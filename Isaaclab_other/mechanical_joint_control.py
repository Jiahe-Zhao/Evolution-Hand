"""Shared action-driven joint targets for mechanical Grasp and Forage."""
import importlib


def layout(raw):
    groups=[]
    for finger in range(5):
        ids=raw.cartesian_ik.joint_ids[finger].tolist()
        if finger:
            ids=[i for i in ids if 'spread' not in raw.hand.joint_names[i]]
        if len(ids)>4:
            raise ValueError('Mechanical joint controller supports four joints per digit')
        groups.append(ids)
    return groups


def encode(raw, targets):
    import torch
    action=torch.zeros((len(targets),20),device=targets.device)
    limits=raw.hand.root_physx_view.get_dof_limits().to(targets.device)
    for finger,ids in enumerate(layout(raw)):
        for slot,joint in enumerate(ids):
            lo,hi=limits[:,joint,0],limits[:,joint,1]
            action[:,4*finger+slot]=(2*(targets[:,joint]-lo)/(hi-lo).clamp_min(1e-6)-1).clamp(-1,1)
    return action


def install(env_id,task):
    if task not in ('grasp','forage'):
        return
    import gymnasium as gym
    import torch
    entry=gym.spec(env_id).entry_point
    module,name=entry.split(':') if isinstance(entry,str) else (None,None)
    cls=getattr(importlib.import_module(module),name) if module else entry
    if getattr(cls,'_mechanical_joint_control_installed',False):
        return
    if task=='forage':
        original_setup=cls._setup_scene
        def setup(self):
            original_setup(self)
            from pxr import Gf,Sdf,UsdPhysics
            # Passive planar support: physics constrains height and rotation;
            # all in-plane motion must still come from actual hand contacts.
            for index in range(self.cfg.scene.num_envs):
                root=f'/World/envs/env_{index}'
                for name,height in (('leaf_one',0.1022),('leaf_two',0.1064)):
                    joint=UsdPhysics.Joint.Define(self.sim.stage,root+'/'+name+'_planar_support')
                    joint.CreateBody0Rel().SetTargets([Sdf.Path(root+'/food')])
                    joint.CreateBody1Rel().SetTargets([Sdf.Path(root+'/'+name)])
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,height-self.cfg.food_cfg.init_state.pos[2]))
                    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
                    joint.CreateExcludeFromArticulationAttr().Set(True)
                    joint.CreateCollisionEnabledAttr().Set(True)
                    for axis in ('transZ','rotX','rotY','rotZ'):
                        limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                        limit.CreateLowAttr().Set(1.0)
                        limit.CreateHighAttr().Set(-1.0)
        cls._setup_scene=setup
    original=cls._apply_action
    def apply(self):
        original_ik=self.cartesian_ik.compute
        def decode(actions,**kwargs):
            limits=self.hand.root_physx_view.get_dof_limits().to(self.device)
            desired=torch.zeros_like(self.prev_targets)
            for finger,ids in enumerate(layout(self)):
                for slot,joint in enumerate(ids):
                    lo,hi=limits[:,joint,0],limits[:,joint,1]
                    desired[:,joint]=lo+(actions[:,4*finger+slot].clamp(-1,1)+1)*0.5*(hi-lo)
            desired=desired.clamp(min=limits[...,0],max=limits[...,1])
            # Compensate the existing target smoother while respecting the
            # final commanded velocity. Reference commands, not lagging pose.
            smoothing=max(float(self.cfg.act_moving_average),1e-6)
            step=0.45*self.sim.get_physics_dt()/smoothing
            return self.prev_targets+(desired-self.prev_targets).clamp(-step,step)
        self.cartesian_ik.compute=decode
        try:
            return original(self)
        finally:
            self.cartesian_ik.compute=original_ik
    cls.mechanical_controller_name=f"{task}_joint_target_v1"
    cls._apply_action=apply
    cls._mechanical_joint_control_installed=True
