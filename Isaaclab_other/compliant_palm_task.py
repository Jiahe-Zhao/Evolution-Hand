"""Single-environment task integration for the experimental FEM palm.

Scene registration keeps physics stepping identical to the task. Reset skinning
initializes nodal coordinates only; subsequent motion is entirely dynamic FEM.
"""
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np


def install_palm_hooks(env_id, asset_root, output_dir, wrist_speed=None, coordinate_roots=False):
    import gymnasium as gym
    import torch
    from isaaclab.assets import DeformableObject, DeformableObjectCfg
    from isaaclab.utils.math import quat_apply
    from compliant_palm_runtime import spawn_compliant_palm
    from audit_mechanical_workspace import graph, fk
    entry = gym.spec(env_id).entry_point
    if isinstance(entry, str):
        module, name = entry.split(':')
        cls = getattr(importlib.import_module(module), name)
    else:
        cls = entry
    original_setup, original_reset, original_step = cls._setup_scene, cls._reset_idx, cls.step
    original_apply = cls._apply_action
    asset_root = Path(asset_root).resolve()
    output_dir = Path(output_dir).resolve()

    def setup(self):
        if self.cfg.scene.num_envs != 1:
            raise ValueError('Experimental palm integration requires one environment; replication not validated')
        original_setup(self)
        urdf = Path(self.cfg.robot_cfg.spawn.asset_path).resolve()
        side = 'left' if urdf.parent.parent.name == 'left' else 'right'
        assets = asset_root/('compliant_palm_left' if side == 'left' else 'compliant_palm')
        hand_path = self.cfg.robot_cfg.prim_path.replace('env_.*', 'env_0')
        palm_path = hand_path.rsplit('/', 1)[0]+'/CompliantPalm'
        binding = spawn_compliant_palm(self.sim.stage, assets, hand_path, palm_path,
            tuple(self.cfg.robot_cfg.init_state.pos), tuple(self.cfg.robot_cfg.init_state.rot))
        self.mechanical_palm = DeformableObject(DeformableObjectCfg(prim_path=palm_path, spawn=None))
        self.scene.deformable_objects['mechanical_palm'] = self.mechanical_palm
        data = np.load(assets/'palm_tetrahedra.npz')
        points = data['points']
        nodes, base = graph(urdf)
        poses = fk(nodes, base, {j.get('name'):0. for j in nodes if j.get('type') != 'fixed'})
        # Linear blend across the four support rays, exact on each support ray.
        u = np.repeat(np.linspace(0, 3, 19), 13*3)
        weights = np.maximum(0., 1.-np.abs(u[:,None]-np.arange(4)[None,:]))
        local = []
        for d in range(2,6):
            inv = np.linalg.inv(poses[f'link_{d}_0'])
            local.append(points@inv[:3,:3].T + inv[:3,3])
        self._palm_local = torch.tensor(np.asarray(local), device=self.device, dtype=torch.float32)
        self._palm_weights = torch.tensor(weights.T, device=self.device, dtype=torch.float32)
        if coordinate_roots:
            from palm_coordination import coordinate_palm_joints
            original_target = self.hand.set_joint_position_target
            original_state = self.hand.write_joint_state_to_sim
            def coordinated_target(target, joint_ids=None, env_ids=None):
                if joint_ids is not None:
                    raise ValueError('Coordinated palm requires full joint target writes')
                return original_target(coordinate_palm_joints(target,list(self.hand.joint_names),self.hand.root_physx_view.get_dof_limits().to(self.device)), env_ids=env_ids)
            def coordinated_state(joint_pos, joint_vel, joint_ids=None, env_ids=None):
                if joint_ids is not None:
                    raise ValueError('Coordinated palm requires full reset state writes')
                return original_state(coordinate_palm_joints(joint_pos,list(self.hand.joint_names),self.hand.root_physx_view.get_dof_limits().to(self.device)), joint_vel, env_ids=env_ids)
            self.hand.set_joint_position_target = coordinated_target
            self.hand.write_joint_state_to_sim = coordinated_state
        self._palm_reset_count = 0
        self._palm_tets = torch.tensor(data['tetrahedra'],device=self.device,dtype=torch.long)
        q = torch.tensor(points,device=self.device,dtype=torch.float32)[self._palm_tets]
        self._palm_rest_inverse = torch.linalg.inv(torch.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],dim=-1))
        self._palm_metrics = {'steps':0,'minimum_volume_ratio':1.,'finite':True,'inversion_detected':False}
        files = [urdf, assets/'palm_tetrahedra.npz', assets/'continuous_palm.stl', assets/'compliant_palm.json', Path(__file__), Path(__file__).with_name('compliant_palm_runtime.py'),Path(__file__).with_name('palm_coordination.py')]
        from palm_coordination import contract as coordination_contract
        contract = {'palm_coordination':coordination_contract() if coordinate_roots else None, 'mechanical_wrist_speed_m_s':wrist_speed, 'side':side,'binding':binding,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    'scope':'Experimental full FEM task evaluation; not a verified training configuration'}
        (output_dir/'palm_assembly_contract.json').write_text(json.dumps(contract,indent=2))

    def reset(self, env_ids):
        original_reset(self, env_ids)
        if not hasattr(self, 'mechanical_palm'):
            return
        self.sim.forward()
        self.hand.update(0.)
        ids = self.hand._ALL_INDICES if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        positions = torch.zeros((len(ids), self._palm_local.shape[1], 3),device=self.device)
        for i, d in enumerate(range(2,6)):
            b = self.hand.body_names.index(f'link_{d}_0')
            q = self.hand.data.body_quat_w[ids,b].unsqueeze(1).expand(-1,positions.shape[1],-1)
            local = self._palm_local[i].unsqueeze(0).expand(len(ids),-1,-1)
            world = quat_apply(q, local)+self.hand.data.body_pos_w[ids,b].unsqueeze(1)
            positions += world*self._palm_weights[i][None,:,None]
        state = torch.cat((positions,torch.zeros_like(positions)),dim=-1)
        q = positions[0][self._palm_tets]
        reset_deformation = torch.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],dim=-1)@self._palm_rest_inverse
        reset_minimum = float(torch.linalg.det(reset_deformation).min())
        np.savez_compressed(output_dir/'palm_last_reset.npz', nodes=positions.detach().cpu().numpy(), joint_names=np.asarray(self.hand.joint_names), joint_positions=self.hand.data.joint_pos.detach().cpu().numpy(), minimum_volume_ratio=reset_minimum)
        print('[PALM_RESET_VOLUME]',reset_minimum,flush=True)
        self.mechanical_palm.write_nodal_state_to_sim(state,env_ids=ids)
        self.mechanical_palm.reset(ids)
        self._palm_reset_count += 1
        print('[PALM_TASK_RESET]',self._palm_reset_count, 'finite',bool(torch.isfinite(state).all()),flush=True)
    def apply(self):
        before = self.hand.data.root_pos_w.clone()
        original_apply(self)
        after = self.hand.data.root_pos_w.clone()
        delta = after-before
        distance = torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
        if self._palm_metrics['steps'] == 0:
            print('[PALM_WRIST_JUMP_M]',distance.detach().cpu().tolist(),flush=True)
        if wrist_speed is not None:
            allowed = float(wrist_speed)*self.sim.get_physics_dt()
            actual = before+delta*torch.clamp(allowed/distance.clamp_min(1e-12),max=1.)
            pose = torch.cat((actual,self.hand.data.root_quat_w.clone()),dim=-1)
            self.hand.write_root_pose_to_sim(pose)
            if hasattr(self,'wrist_offsets'):
                self.wrist_offsets[:] = actual-self.hand.data.default_root_state[:,:3]-self.scene.env_origins

    def step(self, actions):
        result = original_step(self, actions)
        self.mechanical_palm.update(0.)
        x = self.mechanical_palm.data.nodal_pos_w[0]
        finite = bool(torch.isfinite(x).all())
        minimum = None
        if finite:
            q = x[self._palm_tets]
            deformation = torch.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],dim=-1)@self._palm_rest_inverse
            minimum = float(torch.linalg.det(deformation).min())
        metrics = self._palm_metrics
        metrics['steps'] += 1
        metrics['finite'] = metrics['finite'] and finite
        metrics['minimum_volume_ratio'] = min(metrics['minimum_volume_ratio'], minimum) if finite else None
        metrics['inversion_detected'] |= not finite or minimum <= 0
        (output_dir/'palm_runtime_integrity.json').write_text(json.dumps(metrics,indent=2))
        if metrics['inversion_detected']:
            np.savez_compressed(output_dir/'palm_failure.npz',nodes=x.detach().cpu().numpy(),joint_names=np.asarray(self.hand.joint_names),joint_positions=self.hand.data.joint_pos.detach().cpu().numpy(),tetrahedra=self._palm_tets.cpu().numpy())
            raise RuntimeError(f'Physical palm failed during task: {metrics}')
        return result
    cls._setup_scene, cls._reset_idx, cls.step, cls._apply_action = setup, reset, step, apply
