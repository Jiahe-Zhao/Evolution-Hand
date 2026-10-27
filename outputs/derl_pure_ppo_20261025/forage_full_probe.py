"""Full native episode feasibility evidence; never used as BC data."""
import json,os
import torch

def run(env,graph,args,out):
 from four_task_rollout import Teacher
 e=env.unwrapped;c=e.graph_control;t=e.whole_hand_tactile
 teacher=Teacher(e,graph,args.task)
 mode=os.environ.get('CONTACT_PROBE_MODE','teacher')
 if mode=='teacher' and args.task=='forage' and e.num_envs!=1:raise ValueError('Forage teacher is single-environment only')
 if args.task=='branch':
  import four_task_rollout
  four_task_rollout.contact=lambda raw,task:t.digit_forces()
 original=e._get_rewards;rows=[];snapshot={};peaks=[0.,0.];success=torch.zeros(e.num_envs,dtype=torch.bool,device=e.device)
 def capture():
  reward=original();t.update()
  if args.task=='forage':
   forces=t.force[:,:,:2,:].norm(dim=-1).sum(1)
   speeds=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in [e.leaf_one,e.leaf_two]],-1)
   data=dict(dist=e._leaf_distances().tolist(),contact_seen=e.leaf_contact_seen.tolist(),invalid=e.invalid_leaf_motion.tolist(),leaf_z=torch.stack([e.leaf_one.data.root_pos_w[:,2],e.leaf_two.data.root_pos_w[:,2]],-1).tolist(),speed_max=float(speeds.max()))
  else:
   forces=t.digit_forces();data=dict(opposition=t.opposition_cosine().tolist(),hold=e.tactile_streak.tolist(),contact_only_hold=e.tactile_contact_only_streak.tolist(),old_tip_force=e.branch_contact_sensor.data.force_matrix_w.norm(dim=-1).sum(1).tolist(),tip_relative=(e.cartesian_ik.fingertip_positions_world()-e.branch.data.root_pos_w[:,None,:]).tolist(),speed_max=0.)
  snapshot.clear();snapshot.update(data,forces=forces.tolist(),success=(reward>=999).tolist())
  peaks[0]=max(peaks[0],float(forces.max()));peaks[1]=max(peaks[1],data['speed_max'])
  success.logical_or_((reward>=999)&alive)
  return reward
 e._get_rewards=capture
 q=e.hand.data.joint_pos.index_select(-1,c.indices);action=torch.zeros_like(e.graph_actions);action[:,:len(graph.joints)]=2*(q-c.lower)/(c.upper-c.lower)-1
 alive=torch.ones_like(success);length=torch.zeros_like(success,dtype=torch.long)
 for i in range(e.max_episode_length):
  if mode=='teacher' and args.task=='forage' and teacher.stage=='lower':
   force=e.leaf_contact.data.force_matrix_w.norm(dim=-1).sum(1)
   if float(force[0,teacher.leaf])>.05:teacher.stage='sweep'
  a=teacher.action() if mode=='teacher' else action
  obs,reward,term,trunc,info=env.step(a)
  if not torch.isfinite(obs['policy']).all():raise RuntimeError('Nonfinite probe observations')
  length+=alive.long();alive &= ~(term|trunc)
  if i%60==0 or any(snapshot["success"]) or not alive.any():
   rows.append(dict(step=i,stage=teacher.stage,**snapshot))
   (out/'full_probe.json').write_text(json.dumps(dict(mode=mode,task=args.task,steps=i+1,successes=int(success.sum()),num_envs=e.num_envs,peak_force_N=peaks[0],peak_speed_m_s=peaks[1],first_episode_lengths=length.tolist(),trace=rows),indent=2))
  if not alive.any():break
 (out/'completed.json').write_text(json.dumps(dict(status='completed',diagnostic=True,successes=int(success.sum()))))
 print('FULL_PROBE',args.task,mode,int(success.sum()),peaks,flush=True)
