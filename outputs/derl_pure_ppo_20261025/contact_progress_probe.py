"""Full-episode scripted feasibility probe using the exact new reward adapter."""
import json,time
import torch
from native_goal_ppo import save_json

def run(env,graph,args,out):
 from four_task_rollout import Teacher
 e=env.raw;assert env.task=='forage' and e.num_envs==1
 env.reset_seed=args.eval_seed;env.reset();teacher=Teacher(e,graph,'forage')
 trace=[];success=False;total=0.;peak_force=peak_speed=0.;peak_exposure=0.;last={};old=e._get_rewards
 def reward():
  result=old();last.clear();last.update(success=bool(env.event[0]),progress=env.current_progress[0].tolist(),best=env.best_progress[0].tolist(),exposure=float(env.exposure[0]),speed=max(float(a.data.root_lin_vel_w.norm()) for a in (e.leaf_one,e.leaf_two)),force=float(e.whole_hand_tactile.force.norm(dim=-1).max()),leaf_positions=[a.data.root_pos_w[0].tolist() for a in (e.leaf_one,e.leaf_two)],tips=e.cartesian_ik.fingertip_positions_world()[0].tolist())
  return result
 e._get_rewards=reward
 for step in range(e.max_episode_length):
  if teacher.stage=='lower':
   f=e.whole_hand_tactile.force[:,:,teacher.leaf].norm(dim=-1).sum()
   if float(f)>.05:teacher.stage='sweep'
  obs,r,done,info=env.step(teacher.action());total+=float(r[0]);success|=last['success']
  peak_force=max(peak_force,last['force']);peak_speed=max(peak_speed,last['speed']);peak_exposure=max(peak_exposure,last['exposure'])
  if step%60==0 or bool(done[0]):
   trace.append(dict(step=step,stage=teacher.stage,leaf=teacher.leaf,reward=float(r[0]),**last));save_json(out/'probe_progress.json',dict(step=step,stage=teacher.stage,updated=time.time()))
  if bool(done[0]):break
 save_json(out/'probe_report.json',dict(success=success,steps=step+1,raw_return=total,peak_force_N=peak_force,peak_speed_m_s=peak_speed,peak_exposure=peak_exposure,trace=trace))
 save_json(out/'completed.json',dict(status='completed',script_only=True,success=success))
