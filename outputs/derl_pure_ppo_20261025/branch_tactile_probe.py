"""Independent full-episode contact and grasp feasibility probes."""
import json
import torch
from goal_ppo import save_json

def run(env,graph,args,out):
 from four_task_rollout import Teacher
 import four_task_rollout
 e=env.unwrapped;t=e.whole_hand_tactile;c=e.graph_control
 assert args.task=='branch' and e.num_envs==1
 original_contact=four_task_rollout.contact
 four_task_rollout.contact=lambda raw,task:t.digit_forces() if task=='branch' else original_contact(raw,task)
 rows=[];snapshot={};reward=e._get_rewards
 def capture():
  value=reward();t.update()
  digit_vectors=torch.stack([t.force[:,[i for i,g in enumerate(t.groups) if g==d],0].sum(1) for d in t.digits],1)
  thumb=digit_vectors[:,0];other=digit_vectors[:,1:].sum(1);cos=(thumb*other).sum(-1)/(thumb.norm(dim=-1)*other.norm(dim=-1)).clamp_min(1e-8)
  snapshot.update(contact_only_hold=int(e.tactile_contact_only_streak[0]),opposition_cosine=float(cos[0]),success=bool(value[0]>=999),old_success=bool(e.tactile_old_event[0]>=999),digit_force=t.digit_forces()[0].tolist(),body_force=t.force[0,:,0].norm(dim=-1).tolist(),net_force=t.total[0].norm(dim=-1).tolist(),old_tip_force=e.branch_contact_sensor.data.force_matrix_w[0,0].norm(dim=-1).tolist(),hold=int(e.tactile_streak[0]))
  return value
 e._get_rewards=capture
 # Extended holds suppress only early success termination, preserving timeout.
 original_dones=e._get_dones
 def diagnostic_dones():
  term,trunc=original_dones()
  return torch.zeros_like(term),trunc
 e._get_dones=diagnostic_dones
 for trial_index,mode in enumerate(args.probe_modes.split(',')):
  env.reset(seed=args.seed+trial_index)
  def feedback(raw,task):
   f=t.digit_forces()
   # Existing teacher stops at 0.6 N; map to 1.5 N but retain 3 N retreat.
   return torch.where(f>3.,f+1,f*.4) if mode=='teacher_force' else f
  four_task_rollout.contact=feedback
  teacher=Teacher(e,graph,'branch');trace=[]
  initial=dict(branch=e.branch.data.root_state_w[0].tolist(),hand=e.hand.data.root_state_w[0].tolist(),q=e.hand.data.joint_pos[0].tolist(),clearance=getattr(e,'branch_clearance_report',[]))
  start=e.hand.data.joint_pos.clone();peak=torch.zeros(len(t.names),device=e.device);peaknet=peak.clone();success=False;oldsuccess=False;maxlimit=0.
  for step in range(e.max_episode_length):
   if mode=='body_force':
    from isaaclab.utils.math import quat_apply
    t.update();q=e.hand.data.joint_pos.clone();forces=t.digit_forces();jacobians=e.hand.root_physx_view.get_jacobians()
    axis=quat_apply(e.branch.data.root_quat_w,torch.tensor([[0.,0.,1.]],device=e.device))
    for digit in range(1,6):
     candidates=[i for i,name in enumerate(t.names) if name.startswith(f'link_{digit}_') and name.split('_')[-1].isdigit()]
     pos=e.hand.data.body_pos_w[:,candidates];rel=pos-e.branch.data.root_pos_w[:,None,:]
     radial=rel-(rel*axis[:,None,:]).sum(-1,keepdim=True)*axis[:,None,:]
     current=t.force[0,candidates,0].norm(dim=-1)
     chosen=int(current.argmax()) if float(current.max())>.02 else int(radial.norm(dim=-1)[0].argmin())
     body=candidates[chosen];ids=e.cartesian_ik.joint_ids[digit-1]
     direction=-radial[:,chosen]/radial[:,chosen].norm(dim=-1,keepdim=True).clamp_min(1e-6)
     f=forces[:,digit-1];amount=torch.where(f<1.5,torch.full_like(f,.0003),torch.where(f>3.,torch.full_like(f,-.0003),torch.zeros_like(f)))
     error=direction*amount[:,None]
     full=jacobians[:,body-1,:,ids]
     offset=quat_apply(e.hand.data.body_quat_w[:,body],-e.hand.data.body_com_pos_b[:,body])
     jac=full[:,:3,:]+torch.cross(full[:,3:,:].transpose(1,2),offset[:,None,:].expand(-1,len(ids),-1),dim=-1).transpose(1,2)
     delta=(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+1e-5*torch.eye(3,device=e.device)[None],error[:,:,None])).squeeze(-1)
     q[:,ids]+=delta.clamp(-.02,.02)
    values=q.index_select(-1,c.indices).clamp(c.lower,c.upper);action=2*(values-c.lower)/(c.upper-c.lower)-1
   elif mode.startswith('teacher'):action=teacher.action()
   else:
    q=start.clone();desired=.0 if mode=='open_hold' or (mode=='open_then_close' and step<180) else (1.1 if mode=='open_then_close' else int(mode[-2:])/10)
    for j in graph.joints:
     i=e.hand.joint_names.index(j.name)
     if 'spread' not in j.name:q[:,i]=desired
    values=q.index_select(-1,c.indices).clamp(c.lower,c.upper)
    action=2*(values-c.lower)/(c.upper-c.lower)-1
   obs,rew,term,trunc,info=env.step(action)
   peak=torch.maximum(peak,torch.tensor(snapshot['body_force'],device=e.device));peaknet=torch.maximum(peaknet,torch.tensor(snapshot['net_force'],device=e.device))
   q=e.hand.data.joint_pos.index_select(-1,c.indices);maxlimit=max(maxlimit,float(torch.maximum(c.lower-q,q-c.upper).clamp_min(0).max()))
   if step%60==0 or snapshot['success']:trace.append(dict(step=step,**snapshot))
   success|=snapshot['success'];oldsuccess|=snapshot['old_success']
   if (snapshot['hold']>=180 if mode=='open_hold' else success) or bool((term|trunc).any()):break
  rows.append(dict(mode=mode,initial=initial,seed=args.seed+trial_index,sustained_3s=snapshot['hold']>=180,success=success,old_success=oldsuccess,steps=step+1,body_peak_N=dict(zip(t.names,peak.tolist())),net_body_peak_N=dict(zip(t.names,peaknet.tolist())),max_joint_limit_rad=maxlimit,trace=trace))
  save_json(out/'branch_probe.json',dict(scope='script feasibility under unchanged reset, geometry and force thresholds; not PPO success or proof of impossibility',trials=rows))
  print('BRANCH_PROBE',mode,success,float(peak.max()),flush=True)
 # Exercise the expanded PPO observation/reward adapter without optimizer updates.
 from native_goal_adapter import GoalAdapter
 adapter=GoalAdapter(env,'branch',out);observation=adapter.reset()
 q=e.hand.data.joint_pos.index_select(-1,c.indices);neutral=2*(q-c.lower)/(c.upper-c.lower)-1
 next_obs,_,_,_=adapter.step(neutral)
 assert torch.isfinite(next_obs).all()
 save_json(out/'observation_check.json',dict(observation_size=observation.shape[-1],next_observation_size=next_obs.shape[-1],tactile_size=t.features().shape[-1],finite=True))
 save_json(out/'completed.json',dict(status='completed',trials=len(rows),successes=sum(r['success'] for r in rows)))
