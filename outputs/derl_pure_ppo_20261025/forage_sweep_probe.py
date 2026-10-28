"""Bounded 16-path physical sweep search; no policy training or BC export."""
import itertools,time
import torch
from native_goal_ppo import save_json

def run(env,graph,args,out):
 e=env.raw;n=e.num_envs;assert n==16
 cases=list(itertools.product([-.035,-.055],[-.012,-.018,-.024,-.030],[.07,.10]))
 env.reset_seed=args.eval_seed;env.reset();c=e.graph_control
 q=e.hand.data.joint_pos[:,c.indices].clone();neutral=2*(q-c.lower)/(c.upper-c.lower)-1
 z=torch.zeros(n,device=e.device,dtype=torch.bool);alive=~z;success=z.clone();exposure=z.float();force=z.float();speed=z.float();length=z.long();partial=z.float();progress=torch.zeros_like(env.best_progress)
 old=e._get_rewards
 def reward():
  r=old();success.logical_or_(alive&env.event)
  exposure.copy_(torch.maximum(exposure,torch.where(alive,env.exposure,0)))
  force.copy_(torch.maximum(force,torch.where(alive,e.whole_hand_tactile.force.norm(dim=-1).flatten(1).amax(-1),0)))
  v=torch.stack([a.data.root_lin_vel_w.norm(dim=-1) for a in (e.leaf_one,e.leaf_two)],-1).amax(-1)
  speed.copy_(torch.maximum(speed,torch.where(alive,v,0)));partial.add_(r*alive)
  progress.copy_(torch.maximum(progress,torch.where(alive[:,None],env.best_progress,0)))
  return r
 e._get_rewards=reward
 case=torch.tensor(cases,device=e.device);trace=[]
 for step in range(e.max_episode_length):
  a=torch.zeros_like(e.graph_actions);a[:,:len(graph.joints)]=neutral
  wrist=torch.zeros(n,3,device=e.device)
  wrist[:,1]=case[:,0] if step<840 else case[:,2]
  wrist[:,2]=case[:,1] if 600<=step<2160 else 0.
  a[:,len(graph.joints):]=(wrist/e.wrist_limits).clamp(-1,1)
  obs,r,done,info=env.step(a);length.add_(alive);alive&=~done
  if step%120==0:
   trace.append(dict(step=step,exposure=env.exposure.tolist(),progress=env.current_progress.tolist()))
   save_json(out/'probe_progress.json',dict(step=step,successes=int(success.sum()),max_exposure=float(exposure.max()),updated=time.time()))
  if not alive.any():break
 rows=[dict(back_y_m=x[0],depth_z_m=x[1],end_y_m=x[2],success=bool(success[i]),peak_exposure=float(exposure[i]),peak_force_N=float(force[i]),peak_speed_m_s=float(speed[i]),episode_steps=int(length[i]),return_raw=float(partial[i]),best_progress=progress[i].tolist()) for i,x in enumerate(cases)]
 save_json(out/'sweep_report.json',dict(cases=rows,trace=trace,script_only=True));save_json(out/'completed.json',dict(status='completed',successes=int(success.sum()),script_only=True))
