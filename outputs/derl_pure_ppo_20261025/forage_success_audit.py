"""Independent final-policy evaluation with pre-reset success quality evidence."""
import json,os
from pathlib import Path
import torch
from native_goal_ppo import Policy,Moments

def run(env,args,out):
 r=env.raw;n=env.num_envs;checkpoint=Path(os.environ['FORAGE_AUDIT_CHECKPOINT'])
 state=torch.load(checkpoint,map_location=r.device,weights_only=False)
 net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(r.device);net.load_state_dict(state['policy']);net.eval()
 rms=Moments((env.observation_space.shape[0],),r.device)
 for key,value in state['observation_moments'].items():setattr(rms,key,value)
 env.reset_seed=args.eval_seed;obs=env.reset();torch.manual_seed(args.eval_seed+1)
 alive=torch.ones(n,dtype=torch.bool,device=r.device);peak=torch.zeros(n,device=r.device);speed=peak.clone();length=torch.zeros(n,dtype=torch.long,device=r.device);records=[];speed_events=[];over_frames=torch.zeros(n,dtype=torch.long,device=r.device)
 original=r._get_rewards
 def reward():
  rew=original();forces=r.whole_hand_tactile.force[:,:,:2,:].norm(dim=-1)
  current=forces.amax(dim=(1,2));peak[:]=torch.where(alive,torch.maximum(peak,current),peak)
  speeds=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in [r.leaf_one,r.leaf_two]],-1).amax(-1);speed[:]=torch.where(alive,torch.maximum(speed,speeds),speed)
  over_frames.add_((alive&(speeds>2)).long())
  if len(speed_events)<32 and (alive&(speeds>2)).any():
   i=int(torch.where(alive,speeds,torch.zeros_like(speeds)).argmax())
   speed_events.append(dict(env=i,step=int(length[i])+1,speed=float(speeds[i]),force_N=float(current[i]),invalid=bool(r.invalid_leaf_motion[i]),speed_streak=int(r.rl_leaf_speed_streak[i]),terminated=bool(r.reset_terminated[i]),leaf_one=r.leaf_one.data.root_state_w[i].tolist(),leaf_two=r.leaf_two.data.root_state_w[i].tolist(),food=r.food.data.root_state_w[i].tolist()))
  for i in torch.where(alive&env.event)[0].tolist():
   records.append(dict(env=i,steps=int(length[i])+1,peak_single_body_leaf_force_N=float(peak[i]),success_force_N=float(current[i]),peak_leaf_speed_m_s=float(speed[i]),success_leaf_speed_m_s=float(speeds[i]),distances_m=r._leaf_distances()[i].tolist(),contact_seen=r.leaf_contact_seen[i].tolist(),invalid=bool(r.invalid_leaf_motion[i]),leaf_height_m=[float(x.data.root_pos_w[i,2]) for x in [r.leaf_one,r.leaf_two]]))
  return rew
 r._get_rewards=reward
 for step in range(r.max_episode_length):
  with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.sample()
  obs,rew,done,info=env.step(action);length+=alive.long();alive &= ~done
  if not alive.any():break
 report=dict(checkpoint=str(checkpoint),mode='sample',seed=args.eval_seed+1,successes=len(records),num_envs=n,first_episode_lengths=length.tolist(),successful_episodes=records,peak_single_body_leaf_force_N=float(peak.max()),peak_leaf_speed_m_s=float(speed.max()),over_speed_frames=over_frames.tolist(),speed_events=speed_events,threshold_note='Force statistics are diagnostics only; native success unchanged')
 (out/'success_audit.json').write_text(json.dumps(report,indent=2));(out/'completed.json').write_text(json.dumps({'status':'completed'}));print('SUCCESS_AUDIT',len(records),'/',n,flush=True)
