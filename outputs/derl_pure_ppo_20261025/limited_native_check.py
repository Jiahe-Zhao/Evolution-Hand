"""One isolated diagnostic rollout. Demonstration actions never enter PPO/BC."""
import json
import torch

def run(env,graph,args,out):
 from four_task_rollout import Teacher,contact
 e=env.unwrapped
 if e.num_envs!=1:raise ValueError('Limited diagnostic requires one environment')
 teacher=Teacher(e,graph,args.task)
 peak_force=0.;peak_speed=0.;max_limit=0.;seen=None;rows=[]
 control=e.graph_control
 initial=e.hand.data.joint_pos.clone();max_motion=0.
 assets=[e.leaf_one,e.leaf_two] if args.task=='forage' else [getattr(e,'branch' if args.task=='branch' else 'cone')]
 original=e._get_rewards
 def rewards():
  nonlocal peak_force,peak_speed,max_limit,seen,max_motion
  reward=original();f=contact(e,args.task)
  seen=(f>.05) if seen is None else seen|(f>.05)
  peak_force=max(peak_force,float(f.max()))
  peak_speed=max(peak_speed,max(float(x.data.root_lin_vel_w.norm(dim=-1).max()) for x in assets))
  q=e.hand.data.joint_pos.index_select(-1,control.indices)
  max_limit=max(max_limit,float(torch.maximum(control.lower-q,q-control.upper).clamp_min(0).max()))
  max_motion=max(max_motion,float((e.hand.data.joint_pos-initial).abs().max()))
  return reward
 e._get_rewards=rewards
 for t in range(min(e.max_episode_length-1,2400 if args.task=='forage' else 360)):
  action=teacher.action();obs,reward,term,trunc,info=env.step(action)
  if not torch.isfinite(obs['policy']).all():raise RuntimeError('Nonfinite diagnostic observations')
  if t%120==0:rows.append(dict(step=t,stage=teacher.stage,contact_N=contact(e,args.task)[0].cpu().tolist(),wrist=e.graph_wrist_offset[0].cpu().tolist()))
  if args.task=='forage' and seen is not None and bool(seen.any()):break
  if bool((term|trunc).any()):break
 contact_ok=bool(seen.any()) if seen is not None else False
 # Forage specifically needs evidence that its filtered contact channel works.
 passed=max_limit<.05 and max_motion>1e-5 and (contact_ok if args.task=='forage' else True)
 report=dict(task=args.task,passed=passed,steps=t+1,contact_seen=seen.cpu().tolist() if seen is not None else [],peak_force_N=peak_force,peak_object_speed_m_s=peak_speed,max_joint_limit_excess_rad=max_limit,max_joint_motion_rad=max_motion,trace=rows,scope='reach/contact/finite-state check only; not task success or BC')
 (out/'limited_check.json').write_text(json.dumps(report,indent=2));print('LIMITED_CHECK',json.dumps(report),flush=True)
 if not passed:raise RuntimeError('Limited native check failed; inspect limited_check.json')
