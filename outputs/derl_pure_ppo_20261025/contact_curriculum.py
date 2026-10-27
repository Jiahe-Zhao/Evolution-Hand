"""Task-only curriculum. No action overrides, BC, or hand geometry edits."""
import importlib
import os
from types import MethodType
import torch
from native_goal_adapter import GoalAdapter


def install_forage_surface(env_id, cfg):
    """Free leaves on a real surface; food is a flush visual target patch."""
    import gymnasium as gym
    import isaaclab.sim as sim
    module, name = gym.spec(env_id).entry_point.split(':')
    cls = getattr(importlib.import_module(module), name)
    # Keep the original leaf reset heights .1022/.1064; support is at .100 m.
    cfg.leaf_success_hold_steps = 30
    cfg.food_cfg.init_state.pos = (0., 0., .08)
    cfg.food_cfg.spawn.collision_props.collision_enabled = False
    original = cls._setup_scene
    def setup(e):
        original(e)
        table = sim.CuboidCfg(size=(.70, .70, .10),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.16,.19,.22)),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.65,dynamic_friction=.5,restitution=0.))
        for i in range(e.num_envs):
            origin=e.scene.env_origins[i].cpu().tolist()
            table.func(f'/World/envs/env_{i}/forage_table',table,
                       translation=(origin[0],origin[1],.05))
    cls._setup_scene = setup


class CurriculumAdapter(GoalAdapter):
    def __init__(self, env, task, out):
        r=env.unwrapped
        self.stage=int(os.environ.get('STRIKE_TRAIN_STAGE','1'))
        self.phase=torch.ones(r.num_envs,device=r.device,dtype=torch.long)
        if task=='strike' and self.stage==2:
            self.phase[:]=2
            self.phase[::5]=1  # Retain grasp competence during manipulation learning.
        self.initial_tool_z=torch.zeros(r.num_envs,device=r.device)
        self.stage_hold=torch.zeros(r.num_envs,device=r.device,dtype=torch.long)
        self.exposure=torch.zeros(r.num_envs,device=r.device)
        self.stage_successes=0
        reset=r._reset_idx
        def reset_hook(raw,ids):
            reset(ids)
            idx=slice(None) if ids is None else ids
            self.stage_hold[idx]=0
            if task=='strike':self.initial_tool_z[idx]=raw.cone.data.root_pos_w[idx,2]
        r._reset_idx=MethodType(reset_hook,r)
        original=r._get_rewards
        def reward(raw):
            if task=='forage':
                raw._compute_intermediate_values()
                t=raw.whole_hand_tactile;t.update()
                raw.leaf_hand_contact_forces=t.force[:,:,:2].norm(dim=-1).sum(1)
                raw.leaf_contact_seen |= raw.leaf_hand_contact_forces>=raw.cfg.minimum_leaf_contact_force
                raw.interaction_seen=raw.leaf_contact_seen.all(-1)
                self.exposure=self.coverage(raw)
                speeds=torch.stack([a.data.root_lin_vel_w.norm(dim=-1) for a in (raw.leaf_one,raw.leaf_two)],-1)
                distance=raw._leaf_distances()
                # Brief fast contacts reset the completion timer, not the episode.
                raw.invalid_leaf_motion=(~torch.isfinite(speeds).all(-1)) | (distance>.5).any(-1)
                qualified=(self.exposure>=.98)&raw.interaction_seen&(speeds.amax(-1)<.15)&~raw.invalid_leaf_motion
                raw.clear_streak=torch.where(qualified,raw.clear_streak+1,0)
                event=raw.clear_streak>=30
                raw.success_achieved=event.clone()
                raw.success_leaf_distances[event]=distance[event]
                return 1000.*event.float()
            sparse=original()
            if task=='strike':
                forces=raw.tool_finger_contact_forces
                held=(forces[:,0]>=raw.cfg.tool_finger_contact_force_threshold)&((forces[:,1:]>=raw.cfg.tool_finger_contact_force_threshold).sum(-1)>=raw.cfg.min_long_finger_contacts)
                # Lowest cuboid point must clear its initial bottom by 10 mm;
                # center lift alone could credit tilting around the pedestal.
                from isaaclab.utils.math import matrix_from_quat
                rot=matrix_from_quat(raw.cone.data.root_quat_w)
                half=torch.tensor(raw.cone.cfg.spawn.size,device=raw.device)/2
                extent=(rot[:,2,:].abs()*half).sum(-1)
                bottom=raw.cone.data.root_pos_w[:,2]-extent
                lifted=bottom>self.initial_tool_z-half[2]+.010
                stable=(raw.cone.data.root_lin_vel_w.norm(dim=-1)<.15)&(raw.tool_grasp_streak>=raw.cfg.tool_grasp_hold_steps)
                self.stage_hold=torch.where(held&lifted&stable,self.stage_hold+1,0)
                grasp=self.stage_hold>=30
                return torch.where(self.phase==1,1000.*grasp.float(),sparse)
            return sparse
        r._get_rewards=MethodType(reward,r)
        super().__init__(env,task,out)

    @staticmethod
    def coverage(r):
        """Conservative exposure: projected leaf AABBs must uncover target grid.

        AABB can understate exposure for tilted leaves, never credit occluded
        grid samples as exposed. 25x23 cell centers approximate target area.
        """
        from isaaclab.utils.math import matrix_from_quat
        size=r.cfg.food_cfg.spawn.size
        x=(torch.arange(25,device=r.device)+.5)/25-.5
        y=(torch.arange(23,device=r.device)+.5)/23-.5
        grid=torch.cartesian_prod(x*size[0],y*size[1])
        points=r.food.data.root_pos_w[:,None,:2]+grid[None]
        covered=torch.zeros(points.shape[:2],device=r.device,dtype=torch.bool)
        for a in (r.leaf_one,r.leaf_two):
            rotation=matrix_from_quat(a.data.root_quat_w)
            half=torch.tensor(a.cfg.spawn.size,device=r.device)/2
            extent=(rotation[:,:2,:].abs()*half).sum(-1)
            covered |= ((points-a.data.root_pos_w[:,None,:2]).abs()<=extent[:,None]).all(-1)
        return 1-covered.float().mean(-1)

    def features(self):
        extra,phi=super().features()
        r=self.raw
        if self.task=='branch':
            f=r.whole_hand_tactile.digit_forces()
            cosine=r.whole_hand_tactile.opposition_cosine()
            continuous=(f/r.cfg.branch_contact_force_threshold).clamp(0,1)
            opposition=(-cosine).clamp(0,1)*continuous[:,0]*continuous[:,1:].mean(-1)
            phi=phi+opposition
            extra=torch.cat((extra,continuous,cosine[:,None]),-1)
        elif self.task=='forage':
            self.exposure=self.coverage(r)
            phi=self.exposure*2+r.leaf_contact_seen.float().mean(-1)+(r.clear_streak/30).clamp(0,1)
            extra=torch.cat((extra,self.exposure[:,None]),-1)
        else:
            lift=r.cone.data.root_pos_w[:,2]-self.initial_tool_z
            f=(r.tool_finger_contact_forces/r.cfg.tool_finger_contact_force_threshold).clamp(0,1)
            tips=r.cartesian_ik.fingertip_positions_world()
            reach=torch.exp(-(tips-r.cone.data.root_pos_w[:,None]).norm(dim=-1)/.08).mean(-1)
            grasp_phi=reach+f[:,0]+f[:,1:].mean(-1)+self.held.float()*(lift/.015).clamp(0,1)+(self.stage_hold/30).clamp(0,1)
            phi=torch.where(self.phase==1,grasp_phi,phi)
            extra=torch.cat((extra,(self.phase==1).float()[:,None],(self.phase==2).float()[:,None],lift[:,None]/.1,(self.stage_hold/30).clamp(0,1)[:,None]),-1)
        return extra,phi

    def contract(self):
        c=super().contract()
        c.update(version=4,task_curriculum=True,hand_structure_unchanged=True,action_override=False)
        if self.task=='strike':
            c.update(stage=self.stage,grasp_rehearsal_fraction=0. if self.stage==1 else .2,
                stage1='Actual tool: thumb + two long fingers; bottom lifts 10 mm; speed <0.15 m/s; hold 0.5 s',
                stage2='Native held-tool impact success; 20% environments retain stage1 goal',
                success='Current goal completion; stage1 is NOT strike success')
        elif self.task=='forage':
            c.update(support='real static table at z=0.100 m; no leaf joints; flush visual food patch',
                     success='>=98% exposed grid under conservative projected AABB test, both leaves contacted, speed <0.15 m/s for 0.5 s')
        else:c.update(shaping_extra='Continuous finger contact and opposing force feedback; no teacher action stopping')
        return c
