"""Opt-in real-gravity tool support; only reset positions the static fixture."""
import importlib

def install(env_id,cfg):
 import gymnasium as gym
 import torch
 import isaaclab.sim as sim
 from isaaclab.assets import RigidObject,RigidObjectCfg
 module,name=gym.spec(env_id).entry_point.split(':');cls=getattr(importlib.import_module(module),name)
 original_setup=cls._setup_scene;original_reset=cls._reset_idx
 cfg.Cone_cfg.spawn.rigid_props.disable_gravity=False
 cfg.Cone_cfg.spawn.collision_props.contact_offset=.001
 cfg.Cone_cfg.spawn.collision_props.rest_offset=0.
 size=(.012,.012,.020)
 support_cfg=RigidObjectCfg(prim_path='/World/envs/env_.*/ToolSupport',
  spawn=sim.CuboidCfg(size=size,rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
   collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
   physics_material=sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.),
   visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.2,.25,.3))),
  init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,-.5)))
 def setup(self):
  self.tool_support=RigidObject(support_cfg)
  original_setup(self)
  self.scene.rigid_objects['tool_support']=self.tool_support
 def reset(self,env_ids):
  original_reset(self,env_ids)
  ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
  if not ids.numel():return
  from isaaclab.utils.math import matrix_from_quat
  state=self.tool_support.data.default_root_state[ids].clone()
  tool=self.cone.data.root_state_w[ids]
  half=torch.tensor(self.cfg.Cone_cfg.spawn.size,device=self.device)/2
  bottom=tool[:,2]-(matrix_from_quat(tool[:,3:7])[:,2,:].abs()*half).sum(-1)
  state[:,:3]=tool[:,:3];state[:,2]=bottom-size[2]/2
  state[:,7:]=0.
  self.tool_support.write_root_state_to_sim(state,ids)
  self.tool_support_contract={'gravity_enabled':True,'size_m':size,'tool_contact_offset_m':.001,'support_contact_offset_m':.001,'rest_offset_m':0.,'placement':'under actual reset tool, kinematic fixture, no attachment','positions_world':state[:,:3].cpu().tolist()}
 cls._setup_scene=setup;cls._reset_idx=reset

def audit(env):
 import fcl
 import numpy as np
 from scipy.spatial.transform import Rotation
 from imported_hand_geometry import hand_collision_objects
 reports=[]
 for index in range(env.num_envs):
  state=env.tool_support.data.root_state_w[index].cpu().numpy()
  fixture=fcl.CollisionObject(fcl.Box(*env.tool_support_contract['size_m']),fcl.Transform(Rotation.from_quat(state[[4,5,6,3]]).as_matrix(),state[:3]))
  bodies,_=hand_collision_objects(env,index)
  target=env.strike_object.data.root_state_w[index].cpu().numpy()
  bodies.append(('strike_target',fcl.CollisionObject(fcl.Box(*env.cfg.strike_object_cfg.spawn.size),fcl.Transform(Rotation.from_quat(target[[4,5,6,3]]).as_matrix(),target[:3]))))
  hits=[]
  for name,body in bodies:
   result=fcl.CollisionResult();fcl.collide(fixture,body,fcl.CollisionRequest(enable_contact=True,num_max_contacts=16),result)
   if result.is_collision:hits.append({'path':name,'penetration_m':max((float(c.penetration_depth) for c in result.contacts),default=0.)})
  reports.append({'env':index,'fixture_collisions':hits})
 return reports
