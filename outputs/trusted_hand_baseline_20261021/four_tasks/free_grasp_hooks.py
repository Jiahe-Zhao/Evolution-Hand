"""Separate real-gravity Grasp hold diagnostic; not the native fixed-ball scene."""
def install(env_id,cfg,out,task):
 assert task=='grasp'
 import importlib,gymnasium as gym,torch
 import isaaclab.sim as sim
 from isaaclab.assets import RigidObject,RigidObjectCfg
 from baseline_hooks import install as baseline
 cfg.grasp_object_cfg.spawn.rigid_props.kinematic_enabled=False
 cfg.grasp_object_cfg.spawn.rigid_props.disable_gravity=False
 baseline(env_id,cfg,out,task)
 module,name=gym.spec(env_id).entry_point.split(':');cls=getattr(importlib.import_module(module),name);original_setup=cls._setup_scene;original_reset=cls._reset_idx
 support_cfg=RigidObjectCfg(prim_path='/World/envs/env_.*/GraspSupport',spawn=sim.CuboidCfg(size=(.02,.02,.01),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),collision_props=sim.CollisionPropertiesCfg(contact_offset=.0003,rest_offset=0.),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.15,.2,.25))),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-.5)))
 def setup(self):
  original_setup(self);self.grasp_support=RigidObject(support_cfg);self.scene.rigid_objects['grasp_support']=self.grasp_support
 def reset(self,ids):
  original_reset(self,ids)
  state=self.grasp_support.data.default_root_state.clone();state[:,:3]=self.grasp_object.data.root_pos_w;state[:,2]-=self.cfg.grasp_object_radius+.005;state[:,7:]=0;self.grasp_support.write_root_state_to_sim(state)
 cls._setup_scene=setup;cls._reset_idx=reset
