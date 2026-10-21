"""Native-scene feedback scripts, success captured before automatic reset."""
import json,math,hashlib
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from driven_wrist_adapter import hand_state

OBJECTS={'grasp':'grasp_object','branch':'branch','forage':'food','strike':'cone'}

def contact(e,task):
 if task=='strike':return e.tool_finger_contact_forces
 s=e.contact_sensor if task=='grasp' else e.branch_contact_sensor if task=='branch' else e.leaf_contact
 return torch.linalg.vector_norm(s.data.force_matrix_w,dim=-1).sum(dim=1)

from centered_opposition import Teacher

def run(env,graph,out,args,evidence):
 e=env.unwrapped;task=args.task;teacher=Teacher(e,graph,task);obj=getattr(e,OBJECTS[task]);trace=[]
 limits=e.hand.root_physx_view.get_dof_limits().to(e.device);initial=obj.data.root_state_w[0].cpu().tolist()
 def snapshot(reward):
  ok,ev=evidence(task,e,reward);q=e.hand.data.joint_pos;over=torch.maximum(limits[:,:,0]-q,q-limits[:,:,1]).clamp_min(0)
  return dict(success=ok,evidence=ev,joint_pos=q[0].cpu().tolist(),max_joint_excess_rad=float(over.max()),object=obj.data.root_state_w[0].cpu().tolist(),force_N=contact(e,task)[0].cpu().tolist(),wrist=e.graph_wrist_offset[0].cpu().tolist(),stage=teacher.stage,hand_root=hand_state(e)[0].cpu().tolist(),tips=e.cartesian_ik.fingertip_positions_world()[0].cpu().tolist(),tip_goal=getattr(teacher,'tip_goal',torch.zeros((1,3),device=e.device))[0].cpu().tolist(),leaf_index=teacher.leaf,objects={name:getattr(e,name).data.root_state_w[0].cpu().tolist() for name in (('food','leaf_one','leaf_two') if task=='forage' else (OBJECTS[task],))})
 original=e._get_rewards
 def get_rewards():
  r=original();e.baseline_snapshot=snapshot(float(r[0]));return r
 e._get_rewards=get_rewards
 audit={'task':task,'seed':args.seed,'source_model_sha256':graph.urdf_sha256,'controller':'centered_pad_position_orientation_v1','training_paused':True,'scene_alignment':'native task + shared mechanical_scene_contract + baseline_hooks; not deployed to paused training','initial_object':initial,'joint_names':e.hand.joint_names,'dt':e.cfg.sim.dt,'control_dt':e.step_dt}
 for method in ['get_masses','get_material_properties','get_dof_stiffnesses','get_dof_dampings','get_dof_max_velocities','get_contact_offsets']:
  if hasattr(e.hand.root_physx_view,method):audit[method]=getattr(e.hand.root_physx_view,method)().cpu().tolist()
 audit['initial_joint_pos']=e.hand.data.joint_pos[0].cpu().tolist();audit['initial_tips']=e.cartesian_ik.fingertip_positions_world()[0].cpu().tolist();audit['joint_limits']=limits[0].cpu().tolist()
 from startup_geometry import audit as geometry_audit
 geometry_audit(e,task,out)
 (out/'runtime_baseline.json').write_text(json.dumps(audit,indent=2))
 writer=None
 if args.record_video:
  import imageio.v2 as imageio
  from isaacsim.core.utils.viewports import set_camera_view
  center=(e.cartesian_ik.fingertip_positions_world()[0].mean(0)+obj.data.root_pos_w[0])/2
  target=center.cpu().tolist();eye=(center+torch.tensor([.23,-.23,.19],device=e.device)).cpu().tolist()
  set_camera_view(eye=eye,target=target,camera_prim_path='/OmniverseKit_Persp')
  writer=imageio.get_writer(str(out/'validation.mp4'),fps=30,codec='libx264',quality=7)
 success=False;reason='episode time limit';n=min(int(e.max_episode_length)-1,args.max_steps or 10000)
 for i in range(n):
  action=teacher.action();obs,reward,term,trunc,info=env.step(action);row=e.baseline_snapshot;row.update(step=i,time_s=(i+1)*e.step_dt,reward=float(reward[0]));trace.append(row)
  if writer and i%max(1,round(1/(30*e.step_dt)))==0:
   from PIL import Image,ImageDraw
   import numpy as np
   frame=env.render();im=Image.fromarray(frame[:,:,:3]);d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,55),fill=(15,22,32));d.text((12,8),f'BASELINE FULL HAND | {task.upper()} | seed {args.seed} | {row["time_s"]:.2f}s | {teacher.stage}',fill='white');d.text((12,30),f'joint excess {row["max_joint_excess_rad"]:.4f} rad | success {row["success"]}',fill='white');writer.append_data(np.asarray(im))
  if i in (59,179,359):
   from centered_contact_audit import audit
   audit(e,out,i)
  if teacher.free_hold_pass:reason='diagnostic free hold';break
  if row['success']:success=True;reason='native success event';break
  if not bool(torch.isfinite(e.hand.data.joint_pos).all()):reason='nonfinite state';break
  if bool((term|trunc).any()):reason='native termination';break
 if writer:writer.close()
 peak_speed=max(math.sqrt(sum(v*v for v in r['object'][7:10])) for r in trace)
 report=dict(task=task,seed=args.seed,native_success=success,steps=len(trace),reason=reason,final=trace[-1],max_joint_excess_rad=max(r['max_joint_excess_rad'] for r in trace),peak_object_speed_m_s=peak_speed,scope='full-hand native-scene script verification; not policy evaluation or proof of task infeasibility',free_load_validated=teacher.free_hold_pass,centered_grasp=teacher.trace[-1])
 (out/'pad_trace.json').write_text(json.dumps(teacher.trace))
 (out/'trace.json').write_text(json.dumps(trace));(out/'report.json').write_text(json.dumps(report,indent=2));print('FOUR_TASK_REPORT',json.dumps(report),flush=True)
