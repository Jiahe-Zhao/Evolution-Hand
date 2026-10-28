"""Shared Branch task for a baseline mechanical hand and the stock Shadow asset.

Robot differences are restricted to asset, actuator parameters and semantic maps.
No BC, teacher actions, or hand-file changes are used by PPO.
"""
import os,json,math,re,copy,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
import isaaclab.sim as sim
from isaaclab.assets import Articulation,RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse,quat_mul,quat_conjugate
from native_goal_ppo import save_json


def configure(cfg,hand,out):
 cfg.episode_length_s=20.;cfg.reset_dof_pos_noise=.02
 cfg.branch_success_hold_steps=15;cfg.branch_contact_force_threshold=1.;cfg.min_long_finger_contacts=2
 cfg.require_pose_stability=True;cfg.branch_relative_position_tolerance=.012;cfg.branch_relative_rotation_tolerance=.12
 if hand=='shadow':
  from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
  from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
  cfg.robot_cfg=copy.deepcopy(SHADOW_HAND_CFG)
  cfg.robot_cfg.prim_path='/World/envs/env_.*/Robot'
  cfg.robot_cfg.init_state.pos=(0.,0.,.4);cfg.robot_cfg.init_state.rot=(1.,0.,0.,0.)
  cfg.actuated_joint_names=[n for n in ShadowHandEnvCfg().actuated_joint_names if 'WRJ' not in n]
  cfg.fingertip_body_names=['robot0_thdistal','robot0_ffdistal','robot0_mfdistal','robot0_rfdistal','robot0_lfdistal']
  cfg.branch_support_body_names=['robot0_ffproximal','robot0_mfproximal','robot0_lfproximal']
  # Matched angular command rate; preserve native tendon/effort/PD parameters.
  # Keep native hardware limits; .45 rad/s is the shared command slew below.
  if os.environ.get('BRANCH_COMPARISON_FORCE_SPEED_CAP')=='1':cfg.robot_cfg.actuators['fingers'].velocity_limit_sim=.45
 else:
  from joint_graph_control import JointGraph
  g=JointGraph.from_urdf(Path(cfg.robot_cfg.spawn.asset_path))
  cfg.actuated_joint_names=[j.name for j in g.joints]
  cfg.fingertip_body_names=['link_1_3','link_2_2','link_3_2','link_4_2','link_5_2']
 cfg.action_space=len(cfg.actuated_joint_names);cfg.observation_space=4*cfg.action_space+42
 cfg.state_space=0;cfg.events=None
 cfg.robot_cfg.spawn.activate_contact_sensors=True
 cfg.robot_cfg.spawn.rigid_props.max_depenetration_velocity=1.
 cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count=int(os.environ.get('BRANCH_COMPARISON_POSITION_ITERS','8'))
 cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count=int(os.environ.get('BRANCH_COMPARISON_VELOCITY_ITERS','0'))
 cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions=True
 cfg.robot_cfg.spawn.articulation_props.fix_root_link=True
 if hand=='ours':
  cfg.robot_cfg.spawn.usd_dir=str(out/'hand_usd');cfg.robot_cfg.spawn.force_usd_conversion=True
  cfg.robot_cfg.spawn.make_instanceable=False
 cfg.branch_cfg.spawn.collision_props.contact_offset=.0003;cfg.branch_cfg.spawn.collision_props.rest_offset=0.
 cfg.branch_cfg.spawn.physics_material=sim.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)
 cfg.branch_contact_sensor_cfg.filter_prim_paths_expr=['/World/envs/env_.*/Robot/'+n for n in cfg.fingertip_body_names]
 (out/'effective_scene.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
 return cfg


def uniform_materials(e):
 from pxr import Usd,UsdShade,UsdPhysics,PhysxSchema
 root=e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
 if e.variant=='ours':
  from contact_materials import configure as configure_import
  configure_import(e.sim.stage,str(root.GetPath()))
 # Deinstance only the runtime copy so material bindings apply to both assets.
 for _ in range(4):
  instances=[p for p in Usd.PrimRange(root) if p.IsInstance()]
  if not instances:break
  for p in instances:p.SetInstanceable(False)
 mat=UsdShade.Material.Define(e.sim.stage,'/World/BranchComparisonMaterial')
 m=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());m.CreateStaticFrictionAttr(.8);m.CreateDynamicFrictionAttr(.8);m.CreateRestitutionAttr(0.)
 phys=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim());phys.CreateFrictionCombineModeAttr('average');phys.CreateCompliantContactStiffnessAttr(0.);phys.CreateCompliantContactDampingAttr(0.)
 count=0
 for p in Usd.PrimRange(root):
  if p.HasAPI(UsdPhysics.CollisionAPI):
   UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,bindingStrength='strongerThanDescendants',materialPurpose='physics')
   c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0003);c.CreateRestOffsetAttr(0.);count+=1
 save_json(e.out/'material_contract.json',dict(collision_prims=count,friction=.8,restitution=0.,compliance=0.,contact_offset=.0003,common_to_both_hands=True))


class BranchComparisonEnv(DirectRLEnv):
 def __init__(self,cfg,variant,out):
  self.variant=variant;self.out=out;self.placement=None;self.previous_phi=None
  super().__init__(cfg)
  self.indices=torch.tensor([self.hand.joint_names.index(n) for n in cfg.actuated_joint_names],device=self.device)
  lim=self.hand.root_physx_view.get_dof_limits().to(self.device)
  self.lower=lim[0,:,0];self.upper=lim[0,:,1]
  self.graph_control=SimpleNamespace(indices=self.indices,lower=self.lower[self.indices],upper=self.upper[self.indices])
  self.tip_ids=[self.hand.body_names.index(n) for n in cfg.fingertip_body_names]
  self.cartesian_ik=SimpleNamespace(fingertip_positions_world=lambda:self.hand.data.body_pos_w[:,self.tip_ids])
  self.prev_targets=self.hand.data.default_joint_pos.clone();self.actions=torch.zeros((self.num_envs,cfg.action_space),device=self.device)
  self.branch_success_streak=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
  self.tactile_body_groups={}
  for name in self.hand.body_names:
   if variant=='shadow':
    self.tactile_body_groups[name]=next((i for i,p in enumerate(['th','ff','mf','rf','lf'],1) if name.startswith('robot0_'+p)),0)
   else:
    m=re.match(r'link_(\d+)_',name);self.tactile_body_groups[name]=int(m[1]) if m and 0<int(m[1])<=5 else 0
  if variant=='shadow':
   wrist=[self.hand.joint_names.index(n) for n in self.hand.joint_names if 'WRJ' in n]
   lim[:,wrist]=0.;self.hand.root_physx_view.set_dof_limits(lim.cpu(),torch.arange(self.num_envs,dtype=torch.int32))
  self.jitter=.002
  self._reset_idx(torch.arange(self.num_envs,device=self.device))
  save_json(out/'asset_inventory.json',dict(variant=variant,joint_names=self.hand.joint_names,body_names=self.hand.body_names,actuated=cfg.actuated_joint_names,tip_ids=self.tip_ids,joint_limits=lim[0].tolist(),body_pos=(self.hand.data.body_pos_w[0]-self.scene.env_origins[0]).tolist(),body_quat=self.hand.data.body_quat_w[0].tolist(),default_joint_pos=self.hand.data.default_joint_pos[0].tolist(),fixed_base=getattr(self.hand,"is_fixed_base",None)))
 def _setup_scene(self):
  self.hand=Articulation(self.cfg.robot_cfg);self.branch=RigidObject(self.cfg.branch_cfg)
  self.branch_contact_sensor=ContactSensor(self.cfg.branch_contact_sensor_cfg)
  uniform_materials(self)
  self.scene.clone_environments(copy_from_source=False)
  self.scene.articulations['robot']=self.hand;self.scene.rigid_objects['branch']=self.branch;self.scene.sensors['branch_contact_sensor']=self.branch_contact_sensor
  light=sim.DomeLightCfg(intensity=1500.,color=(.7,.7,.7));light.func('/World/Light',light)
 def _pre_physics_step(self,actions):self.actions=actions.clone()
 def _apply_action(self):
  c=self.graph_control
  target=c.lower+(self.actions.clamp(-1,1)+1)*.5*(c.upper-c.lower)
  old=self.prev_targets[:,self.indices];target=old+(target-old).clamp(-.45*self.cfg.sim.dt,.45*self.cfg.sim.dt)
  self.prev_targets[:,self.indices]=target;self.hand.set_joint_position_target(self.prev_targets)
 def _compute_intermediate_values(self):pass
 def _branch_relative_pose(self):
  q=self.hand.data.root_quat_w
  return quat_apply_inverse(q,self.branch.data.root_pos_w-self.hand.data.root_pos_w),quat_mul(quat_conjugate(q),self.branch.data.root_quat_w)
 def _get_rewards(self):return torch.zeros(self.num_envs,device=self.device)
 def _get_dones(self):return self.branch_success_streak>=15,self.episode_length_buf>=self.max_episode_length-1
 def _get_observations(self):
  c=self.graph_control;norm=lambda x:2*(x[:,self.indices]-c.lower)/(c.upper-c.lower).clamp_min(1e-6)-1
  q=norm(self.hand.data.joint_pos);dq=self.hand.data.joint_vel[:,self.indices]/.45;command=norm(self.prev_targets)
  hand=self.hand.data.root_state_w.clone();hand[:,:3]-=self.scene.env_origins
  branch=self.branch.data.root_state_w.clone();branch[:,:3]-=self.scene.env_origins
  tips=(self.hand.data.body_pos_w[:,self.tip_ids]-self.scene.env_origins[:,None]).flatten(1)
  return {'policy':torch.cat((q,dq,command,self.actions,hand,branch,tips,self.episode_length_buf[:,None]*self.step_dt),-1)}
 def _reset_idx(self,ids):
  if ids is None:ids=torch.arange(self.num_envs,device=self.device)
  super()._reset_idx(ids)
  if not hasattr(self,'indices'):return
  q=self.hand.data.default_joint_pos[ids].clone()
  q[:,self.indices]+=(torch.rand(len(ids),len(self.indices),device=self.device)*2-1)*self.cfg.reset_dof_pos_noise*(self.upper[self.indices]-self.lower[self.indices])
  q=q.clamp(self.lower,self.upper);v=torch.zeros_like(q)
  self.hand.write_joint_state_to_sim(q,v,env_ids=ids);self.prev_targets[ids]=q;self.actions[ids]=0
  self.hand.set_joint_position_target(q,env_ids=ids);self.sim.forward();self.scene.update(0.)
  state=self.branch.data.default_root_state[ids].clone();state[:,:3]+=self.scene.env_origins[ids]
  # Inspection starts with an explicitly separated branch. Training requires a manifest.
  if self.placement is None:state[:,:3]=self.scene.env_origins[ids]+torch.tensor([0.,0.,.1],device=self.device)
  else:
   state[:,:3]=self.scene.env_origins[ids]+torch.tensor(self.placement['center_world'],device=self.device)+(torch.rand(len(ids),3,device=self.device)*2-1)*self.jitter
   state[:,3:7]=torch.tensor(self.placement['quat_world'],device=self.device)
  state[:,7:]=0;self.branch.write_root_state_to_sim(state,env_ids=ids);self.branch_success_streak[ids]=0


def run(cfg,args,out):
 variant=os.environ['BRANCH_COMPARISON_HAND'];mode=os.environ.get('BRANCH_COMPARISON_MODE','inspect')
 if mode=='native_smoke':
  from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg
  from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv as ShadowHandEnv
  native=ShadowHandEnvCfg();native.scene.num_envs=4;native.events=None;native.seed=args.seed
  env=ShadowHandEnv(native);env.reset();torch.manual_seed(args.seed);peak=0.
  for step in range(1200):
   env.step(torch.rand(4,20,device=env.device)*2-1)
   if not torch.isfinite(env.hand.data.joint_pos).all():raise RuntimeError('Official task nonfinite state')
   peak=max(peak,float(env.hand.data.joint_vel.abs().max()))
  save_json(out/'completed.json',dict(status='completed',official_task_smoke_only=True,steps=1200,finite=True,peak_joint_velocity=peak))
  return
 cfg=configure(cfg,variant,out)
 if mode in ('inspect','kinematics','replay'):cfg.reset_dof_pos_noise=0.
 e=BranchComparisonEnv(cfg,variant,out)
 from whole_hand_tactile import attach
 attach(e,'branch',out,branch_success=True)
 if mode=='kinematics':
  from branch_comparison_geometry import export_kinematics,CollisionCache,verify_cache,anatomical_frame
  save_json(out/'kinematics.json',export_kinematics(e))
  save_json(out/'anatomical_frame.json',anatomical_frame(e))
  save_json(out/'collision_cache_validation.json',verify_cache(e,CollisionCache(e)))
  save_json(out/'completed.json',dict(status='completed',inspection_only=True))
  return
 if mode=='inspect':
  from branch_comparison_geometry import anatomical_frame,CollisionCache
  save_json(out/'anatomical_frame.json',anatomical_frame(e))
  cache=CollisionCache(e)
  save_json(out/'collision_cache.json',dict(shape_count=len(cache.items)))
  q=e.hand.data.joint_pos[:,e.indices];a=2*(q-e.graph_control.lower)/(e.graph_control.upper-e.graph_control.lower)-1
  initial=e.hand.data.joint_pos.clone();max_force=0.
  for i in range(120):
   e.step(a);max_force=max(max_force,float(e.whole_hand_tactile.digit_forces().max()))
  save_json(out/'completed.json',dict(status='completed',inspection_only=True,finite=bool(torch.isfinite(e.hand.data.joint_pos).all()),max_joint_drift=float((e.hand.data.joint_pos-initial).abs().max()),max_branch_force=max_force))
  return
 from branch_comparison_protocol import install, replay, train
 install(e,out,mode)
 if mode in ('replay','validate'):return replay(e,args,out,mode)
 if mode=='train':return train(e,args,out)
 raise NotImplementedError(mode)
