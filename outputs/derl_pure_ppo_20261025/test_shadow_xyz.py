"""Small physical-only neutral/XYZ test; never loads or stops the training job."""
import sys,json,copy
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import torch
from isaaclab.utils.math import quat_apply,quat_apply_inverse
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
from shadow_xyz_wrapper import build,NAMES
b=Path(__file__).resolve().parent;out=b/'shadow_xyz_20261029';out.mkdir(exist_ok=True)
source=json.loads((out/'stock_inventory.json').read_text())['source'];path=out/'shadow_xyz.usda'
if not path.exists():build(source,path)
ctx=sim.SimulationContext(sim.SimulationCfg(dt=1/240,device='cuda:0'))
cfg=copy.deepcopy(SHADOW_HAND_CFG);cfg.prim_path='/World/Shadow';cfg.spawn.usd_path=str(path);cfg.spawn.articulation_props.fix_root_link=True;cfg.init_state.pos=(0,0,.5);cfg.init_state.rot=(1,0,0,0)
cfg.spawn.articulation_props.solver_velocity_iteration_count=0;cfg.spawn.articulation_props.solver_position_iteration_count=8
cfg.init_state.joint_pos={'.*':0.}
cfg.init_state.joint_vel={'.*':0.}
cfg.actuators['xyz']=ImplicitActuatorCfg(joint_names_expr=NAMES,effort_limit_sim=100.,velocity_limit_sim=.02,stiffness=3000.,damping=100.,armature=0.)
hand=Articulation(cfg);ctx.reset();ids=[hand.joint_names.index(n) for n in NAMES];mount=hand.body_names.index('robot0_hand_mount');q=hand.data.default_joint_pos.clone();hand.write_joint_state_to_sim(q,torch.zeros_like(q));hand.set_joint_position_target(q);ctx.step();hand.update(1/240);initial=hand.data.body_pos_w[:,mount].clone();rows=[];reference_quat=hand.data.body_quat_w[:,mount].clone()
for label,axis in [('static',None),('local_x',0),('local_y',1),('local_z',2),('world_x',0),('world_y',1),('world_z',2)]:
 start=hand.data.joint_pos[:,ids].clone();startpos=hand.data.body_pos_w[:,mount].clone();peak=0.
 direction=torch.zeros((1,3),device=q.device)
 if axis is not None:direction[:,axis]=1
 local=quat_apply_inverse(reference_quat,direction) if label.startswith('world_') else direction
 expected=direction[0]*.01 if label.startswith('world_') else quat_apply(reference_quat,direction)[0]*.01
 for t in range(480):
  if axis is not None:q[:,ids]=start+local*min((t+1)/240*.006,.01)
  hand.set_joint_position_target(q);hand.write_data_to_sim();ctx.step();hand.update(1/240)
  if not torch.isfinite(hand.data.body_state_w).all():raise RuntimeError('Nonfinite Shadow state')
  peak=max(peak,float(hand.data.body_lin_vel_w.norm(dim=-1).max()))
 delta=(hand.data.body_pos_w[:,mount]-startpos)[0]
 rows.append(dict(mode=label,mount_displacement_m=delta.tolist(),tracking_error_m=float((delta-expected).norm()),peak_body_speed_m_s=peak,stage_positions_m=hand.data.joint_pos[:,ids].tolist()))
 (out/'wrist_test.json').write_text(json.dumps(dict(rows=rows,reference_mount_quaternion_wxyz=reference_quat.tolist(),joint_names=hand.joint_names,body_names=hand.body_names),indent=2))
(out/'wrist_test_completed.json').write_text(json.dumps(dict(passed=all(r['tracking_error_m']<.002 and r['peak_body_speed_m_s']<1 for r in rows),rows=rows),indent=2))
app.close()
