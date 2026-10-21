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

class Teacher:
 def __init__(self,e,graph,task):
  self.e=e;self.graph=graph;self.task=task;self.c=e.graph_control
  self.q0=e.hand.data.joint_pos.clone();self.wrist=e.graph_wrist_offset.clone();self.stage='contact';self.leaf=1;self.clear=0
  self.direction=None;self.hold=0;self.anchor=None;self.start=getattr(e,OBJECTS[task]).data.root_pos_w.clone()
 def action(self):
  e=self.e;c=self.c;e._compute_intermediate_values();task=self.task
  tips=e.cartesian_ik.fingertip_positions_world();obj=getattr(e,OBJECTS[task]);q=e.hand.data.joint_pos.clone()
  f=contact(e,task)
  if task=='forage':
   leaf=e.leaf_two if self.leaf else e.leaf_one;pos=leaf.data.root_pos_w;away=pos[:,:2]-e.food.data.root_pos_w[:,:2];away/=away.norm(dim=-1,keepdim=True).clamp_min(1e-6)
   finger=1 if self.leaf else 3;tip=tips[:,finger];dist=float(e._leaf_distances()[0,self.leaf]);target=pos.clone()
   cfg=e.cfg.leaf_two_cfg if self.leaf else e.cfg.leaf_one_cfg
   direction=torch.cat((away,torch.zeros_like(away[:,:1])),dim=-1)
   local=quat_apply_inverse(leaf.data.root_quat_w,direction);half=torch.tensor(cfg.spawn.size,device=e.device)/2
   radius=torch.where(local.abs()>1e-6,half/local.abs().clamp_min(1e-6),torch.full_like(local,float('inf'))).min(-1).values
   target[:,:2]-=away*(radius[:,None]-.002)
   if self.stage=='contact':
    target[:,:2]-=away*.006;target[:,2]+=.025
    if float((target-tip).norm())<.005:self.stage='lower'
   elif self.stage=='lower':
    if float((target-tip).norm())<.004:self.stage='sweep'
   elif self.stage=='sweep':
    target=tip.clone();target[:,:2]+=away*.003
    if dist>=e.cfg.leaf_clear_distance+.005:self.stage='lift'
   elif self.stage=='lift':
    target=tip.clone();target[:,2]=pos[:,2]+.025
    if float(tip[0,2]-pos[0,2])>.020 and self.leaf==1:self.leaf=0;self.stage='contact'
   self.tip_goal=target.clone();self.active_finger=finger
   self.wrist=e.graph_wrist_offset+(target-tip).clamp(-.003,.003)
   q=self.q0.clone()
  else:
   center=obj.data.root_pos_w;rot=obj.data.root_quat_w[:,None,:].expand(-1,5,-1)
   local=quat_apply_inverse(rot,tips-center[:,None,:])
   if self.direction is None:
    self.direction=local.clone()
    if task!='grasp':self.direction[:,:,2]=0
    self.direction/=self.direction.norm(dim=-1,keepdim=True).clamp_min(1e-6)
   if task=='grasp':target_local=self.direction*(e.cfg.grasp_object_radius-.0005)
   elif task=='branch':
    target_local=local.clone();target_local[:,:,:2]=self.direction[:,:,:2]*(e.cfg.branch_radius-.0005)
   else:
    half=torch.tensor(e.cfg.Cone_cfg.spawn.size,device=e.device)/2
    target_local=local.clone();radius=half[:2]/self.direction[:,:,:2].abs().clamp_min(1e-6)
    target_local[:,:,:2]=self.direction[:,:,:2]*(radius.min(-1).values[:,:,None]-.0005)
    target_local[:,:,2]=local[:,:,2].clamp(-float(half[2])+.003,float(half[2])-.003)
   targets=center[:,None,:]+quat_apply(rot,target_local)
   self.tip_goal=targets.clone()
   error=targets-tips
   # Hold contacted digits; retreat gently when force exceeds 3 N.
   error=torch.where((f>.6)[:,:,None],torch.zeros_like(error),error)
   outward=quat_apply(rot,self.direction)
   error=torch.where((f>3.)[:,:,None],outward*.0005,error)
   jacobians=e.hand.root_physx_view.get_jacobians();ik=e.cartesian_ik
   for finger,(body,ids) in enumerate(zip(ik.body_ids,ik.joint_ids)):
    if not len(ids):continue
    jac=ik._finger_jacobian_world(body,ids,jacobians);err=error[:,finger];err*=torch.clamp(.001/err.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1.)
    eye=torch.eye(3,device=e.device)[None];delta=(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+.00001*eye,err[:,:,None])).squeeze(-1)
    q[:,ids]+=delta.clamp(-.02,.02)
   if task=='strike':
    held=bool(e.tool_was_held[0]);self.hold=self.hold+1 if held else 0
    if self.stage=='contact' and self.hold>=40:self.stage='lift';self.anchor=obj.data.root_pos_w.clone()
    if self.stage=='lift':
     self.wrist[:,2]=.01
     if float(obj.data.root_pos_w[0,2]-self.anchor[0,2])>.007:self.stage='carry'
    if self.stage=='carry':
     self.wrist[:,:2]=e.strike_object.data.root_pos_w[:,:2]-self.start[:,:2]
     if float((obj.data.root_pos_w[:,:2]-e.strike_object.data.root_pos_w[:,:2]).norm())<.004:self.stage='strike'
    if self.stage=='strike':self.wrist[:,2]=-.095
    if self.stage!='contact' and not held:self.stage='lost_grasp';self.wrist=e.graph_wrist_offset.clone()
  desired=q.index_select(-1,c.indices).clamp(c.lower,c.upper)
  action=torch.zeros_like(e.graph_actions);action[:,:len(self.graph.joints)]=2*(desired-c.lower)/(c.upper-c.lower)-1
  if task in ('forage','strike'):
   scale=e.wrist_limits if task=='forage' else e.wrist_action_scale
   action[:,len(self.graph.joints):]=(self.wrist/scale).clamp(-1,1)
  return action

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
 audit={'task':task,'seed':args.seed,'source_model_sha256':graph.urdf_sha256,'controller':'surface_jacobian_and_contact_feedback_v1','training_paused':True,'scene_alignment':'native task + shared mechanical_scene_contract + baseline_hooks; not deployed to paused training','initial_object':initial,'joint_names':e.hand.joint_names,'dt':e.cfg.sim.dt,'control_dt':e.step_dt}
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
 released=False;anchor=None;release_step=None;max_slip=0.;native_contact=False
 success=False;reason='episode time limit';n=min(int(e.max_episode_length)-1,args.max_steps or 10000)
 for i in range(n):
  action=teacher.action();obs,reward,term,trunc,info=env.step(action);row=e.baseline_snapshot;row.update(step=i,time_s=(i+1)*e.step_dt,reward=float(reward[0]));trace.append(row)
  if writer and i%max(1,round(1/(30*e.step_dt)))==0:
   from PIL import Image,ImageDraw
   import numpy as np
   frame=env.render();im=Image.fromarray(frame[:,:,:3]);d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,55),fill=(15,22,32));d.text((12,8),f'BASELINE FULL HAND | {task.upper()} | seed {args.seed} | {row["time_s"]:.2f}s | {teacher.stage}',fill='white');d.text((12,30),f'joint excess {row["max_joint_excess_rad"]:.4f} rad | success {row["success"]}',fill='white');writer.append_data(np.asarray(im))
  if row['success'] and not released:
   native_contact=True;released=True;release_step=i;anchor=torch.tensor(row['object'][:3],device=e.device)
   state=e.grasp_support.data.root_state_w.clone();state[:,2]-=.1;e.grasp_support.write_root_state_to_sim(state)
  if released:
   slip=float((torch.tensor(row['object'][:3],device=e.device)-anchor).norm());max_slip=max(max_slip,slip)
   if slip>.003:reason='free ball slipped more than 3 mm';break
   if (i-release_step)*e.step_dt>=3.:success=True;reason='3 s free-ball static hold';break
  if not bool(torch.isfinite(e.hand.data.joint_pos).all()):reason='nonfinite state';break
  if bool((term|trunc).any()):reason='native termination';break
 if writer:writer.close()
 peak_speed=max(math.sqrt(sum(v*v for v in r['object'][7:10])) for r in trace)
 report=dict(task=task,seed=args.seed,native_success=native_contact,free_hold_success=success,released=released,max_slip_m=max_slip,steps=len(trace),reason=reason,final=trace[-1],max_joint_excess_rad=max(r['max_joint_excess_rad'] for r in trace),peak_object_speed_m_s=peak_speed,scope='full-hand native-scene script verification; not policy evaluation or proof of task infeasibility',free_load_validated=success)
 (out/'trace.json').write_text(json.dumps(trace));(out/'report.json').write_text(json.dumps(report,indent=2));print('FOUR_TASK_REPORT',json.dumps(report),flush=True)
