"""Baseline contact config shared by validation and future training setup."""
def install(env_id,cfg,out,task):
 import gymnasium as gym,importlib,json
 from pathlib import Path
 from pxr import Usd,UsdShade
 import isaaclab.sim as sim
 from contact_materials import configure
 module,name=gym.spec(env_id).entry_point.split(':');cls=getattr(importlib.import_module(module),name)
 import os
 if os.environ.get('BASELINE_RL_TOLERANCE')=='1':
  from rl_tolerance import install as tolerance
  tolerance(cls,cfg,task,out)
 original=cls._setup_scene
 cfg.robot_cfg.spawn.make_instanceable=False
 cfg.robot_cfg.spawn.usd_dir=str(Path(out).resolve()/'hand_usd');cfg.robot_cfg.spawn.force_usd_conversion=True
 def setup(self):
  original(self)
  root=self.hand.cfg.prim_path.replace('.*','0')
  rows=[]
  for index in range(cfg.scene.num_envs):
   rows.extend(configure(self.sim.stage,self.hand.cfg.prim_path.replace('.*',str(index))))
  (out/'baseline_contact_shapes.json').write_text(json.dumps(rows,indent=2))
  # Appearance only; physical material-purpose bindings remain separate.
  mats={}
  for role,color in {'pad':(.06,.55,.30),'metal':(.55,.30,.08)}.items():
   p='/World/BaselineAppearance/'+role;c=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.55);c.func(p,c);mats[role]=UsdShade.Material(self.sim.stage.GetPrimAtPath(p))
  for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(root)):
   if prim.GetTypeName()=='Mesh':UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats['pad' if 'finger_pad' in str(prim.GetPath()) else 'metal'],bindingStrength='strongerThanDescendants')
  for prim in self.sim.stage.Traverse():
   if prim.GetTypeName()=='DomeLight':
    from pxr import Sdf
    prim.CreateAttribute('inputs:visibleInPrimaryRay',Sdf.ValueTypeNames.Bool).Set(False)
  (out/'native_physics_config.json').write_text(json.dumps(self.cfg.to_dict(),default=str,indent=2))
 cls._setup_scene=setup
