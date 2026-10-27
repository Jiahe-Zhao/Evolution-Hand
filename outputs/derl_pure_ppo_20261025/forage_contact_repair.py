"""Explicit per-body contact mapping and declared planar leaf support."""
import importlib,json
from types import SimpleNamespace

def install(env_id,cfg,out):
 import gymnasium as gym
 # Keep a valid one-body sensor during construction; replace public readings
 # with per-body full-hand views after initialization, before first reset.
 cfg.leaf_one_contact_cfg.prim_path=cfg.leaf_one_cfg.prim_path
 cfg.leaf_one_contact_cfg.filter_prim_paths_expr=[cfg.leaf_one_cfg.prim_path,cfg.leaf_two_cfg.prim_path]
 module,name=gym.spec(env_id).entry_point.split(':');module=importlib.import_module(module);cls=getattr(module,name)
 # Native setup rewrites the sensor prim_path to Robot/.*. Intercept its
 # constructor after that rewrite, otherwise batch initialization still fails.
 sensor_class=module.ContactSensor
 def make_sensor(sensor_cfg):
  if any('leaf_one' in path for path in sensor_cfg.filter_prim_paths_expr):
   sensor_cfg.prim_path=cfg.leaf_one_cfg.prim_path
   sensor_cfg.filter_prim_paths_expr=[cfg.leaf_one_cfg.prim_path,cfg.leaf_two_cfg.prim_path]
  return sensor_class(sensor_cfg)
 module.ContactSensor=make_sensor
 original=cls._setup_scene
 def setup(e):
  original(e)
  from pxr import Gf,Sdf,UsdPhysics
  for index in range(e.num_envs):
   root=f'/World/envs/env_{index}'
   for name,asset in [('leaf_one',cfg.leaf_one_cfg),('leaf_two',cfg.leaf_two_cfg)]:
    joint=UsdPhysics.Joint.Define(e.sim.stage,root+'/'+name+'_planar_support')
    joint.CreateBody0Rel().SetTargets([Sdf.Path(root+'/food')]);joint.CreateBody1Rel().SetTargets([Sdf.Path(root+'/'+name)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,asset.init_state.pos[2]-cfg.food_cfg.init_state.pos[2]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateExcludeFromArticulationAttr().Set(True);joint.CreateCollisionEnabledAttr().Set(True)
    for axis in ['transZ','rotX','rotY','rotZ']:
     limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis);limit.CreateLowAttr().Set(1.);limit.CreateHighAttr().Set(-1.)
 cls._setup_scene=setup

class ContactProxy:
 def __init__(self,t):self.t=t
 @property
 def data(self):
  self.t.update()
  return SimpleNamespace(force_matrix_w=self.t.force[:,:,:2,:],net_forces_w=self.t.total)

def attach(e,out):
 from whole_hand_tactile import attach as attach_tactile
 t=attach_tactile(e,'forage',out)
 e.legacy_leaf_contact=e.leaf_contact;e.leaf_contact=ContactProxy(t)
 paths=[f'/World/envs/env_{i}/{leaf}_planar_support' for i in range(e.num_envs) for leaf in ['leaf_one','leaf_two']]
 if not all(e.sim.stage.GetPrimAtPath(p).IsValid() for p in paths):raise RuntimeError('Missing declared leaf planar support')
 (out/'forage_repair_contract.json').write_text(json.dumps(dict(contact='one view per hand body, two leaf channels plus food tactile',planar_support_joints=len(paths),success_thresholds_unchanged=True,hand_structure_unchanged=True),indent=2))
 return t
