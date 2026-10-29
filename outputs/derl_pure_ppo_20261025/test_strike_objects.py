"""Isolate native free-tool/support physics before adding hand contacts."""
import json
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import torch
import isaaclab.sim as sim
from isaaclab.assets import RigidObject,RigidObjectCfg
from strike_shared_scene import load_contract,rigid_object_from_contract
b=Path(__file__).resolve().parent;out=b/'shadow_xyz_20261029'
c=load_contract(b/'strike_50m_pair_20261029/ours/native_physics_config.json')
dt=c['sim_dt'];ctx=sim.SimulationContext(sim.SimulationCfg(dt=dt,device='cuda:0'))
toolcfg=rigid_object_from_contract(c['Cone_cfg'],'/World/Tool');toolcfg.init_state.pos=(0,0,.35);toolcfg.init_state.rot=(1,0,0,0)
tool=RigidObject(toolcfg);target=RigidObject(rigid_object_from_contract(c['strike_object_cfg'],'/World/Target'))
s=c['support'];bottom=.35-c['Cone_cfg']['spawn']['size'][2]/2
support=RigidObject(RigidObjectCfg(prim_path='/World/Support',spawn=sim.CuboidCfg(size=tuple(s['size']),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),collision_props=sim.CollisionPropertiesCfg(contact_offset=s['contact_offset'],rest_offset=s['rest_offset']),physics_material=sim.RigidBodyMaterialCfg(static_friction=s['static_friction'],dynamic_friction=s['dynamic_friction'])),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,bottom-s['size'][2]/2))))
ctx.reset();rows=[]
for label,dx in [('centered',0.),('offset_2mm',.002)]:
 state=tool.data.default_root_state.clone();state[:,:3]=torch.tensor([dx,0,.35],device=state.device);state[:,3:7]=torch.tensor([1.,0,0,0],device=state.device);state[:,7:]=0;tool.write_root_state_to_sim(state);tool.reset()
 peak=0.;finite=True;trace=[]
 for t in range(round(5/dt)):
  ctx.step();tool.update(dt);target.update(dt);support.update(dt)
  finite=finite and bool(torch.isfinite(tool.data.root_state_w).all())
  peak=max(peak,float(tool.data.root_lin_vel_w.norm(dim=-1).max()))
  if t%max(1,round(.1/dt))==0:trace.append(tool.data.root_state_w[0].tolist())
 rows.append(dict(case=label,finite=finite,peak_speed=peak,final_state=tool.data.root_state_w[0].tolist(),trace=trace))
(out/'object_support_test.json').write_text(json.dumps(dict(scope='Native tool and support only; no hand present, no contact/grasp qualification',dt=dt,rows=rows),indent=2));app.close()
