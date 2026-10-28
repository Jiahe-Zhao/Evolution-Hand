"""Frozen checkpoint, matched-scene full-episode exploration audit."""
import os,json
import torch
from types import MethodType
from native_goal_ppo import Policy,Moments,save_json

def run(env,args,out):
 checkpoint=os.environ['CONTACT_POLICY_AUDIT']
 obs=env.reset();net=Policy(obs.shape[-1],env.action_space.shape[0]).to(obs.device)
 ck=torch.load(checkpoint,map_location=obs.device,weights_only=False);net.load_state_dict(ck['policy']);net.eval()
 rms=Moments((obs.shape[-1],),obs.device);state=ck['observation_moments']
 rms.mean=state['mean'];rms.var=state['var'];rms.count=state['count']
 # Capture terminal progress before automatic reset overwrites state.
 from forage_state_audit import StateAudit
 state_audit=StateAudit(env.raw,out) if env.task=="forage" else None
 original=env.task_reward
 def reward(self,*a):
  result=original(*a);self.audit_progress=self.current_progress.clone()
  if state_audit is not None and hasattr(self,"audit_alive"):state_audit.sample(self.audit_alive)
  if self.task=='forage' and hasattr(self,'audit_alive'):
   r=self.raw;speed=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in (r.leaf_one,r.leaf_two)],-1)
   ids=((speed.amax(-1)>2)&self.audit_alive).nonzero().flatten()
   if len(ids) and len(self.speed_events)<30:
    i=int(ids[0]);self.speed_events.append(dict(step=self.audit_step,env=i,speeds=speed[i].tolist(),positions=[(x.data.root_pos_w[i]-r.scene.env_origins[i]).tolist() for x in (r.leaf_one,r.leaf_two)],force=float(r.whole_hand_tactile.force[i].norm(dim=-1).max()),success=bool(self.event[i])))
  return result
 env.task_reward=MethodType(reward,env)
 results=[]
 modes=[('mean',0.,1),('sample',1.,1),('quarter_noise',.25,1),('held_noise_12',1.,12)]
 if os.environ.get('CONTACT_AUDIT_MEAN_ONLY'):modes=modes[:1]
 if os.environ.get('CONTACT_AUDIT_MODE'):modes=[x for x in modes if x[0]==os.environ['CONTACT_AUDIT_MODE']]
 for name,scale,period in modes:
  env.speed_events=[]
  env.reset_seed=int(os.environ.get('CONTACT_AUDIT_SCENE_SEED','1062802'));obs=env.reset();torch.manual_seed(1062803)
  n=env.num_envs;alive=torch.ones(n,dtype=torch.bool,device=obs.device);passed=~alive
  peak=torch.zeros(n,len(env.progress_names),device=obs.device);ret=torch.zeros(n,device=obs.device)
  count=clip=reverse=zero=latezero=latecount=0.;prev=None;lastdelta=None
  for step in range(args.episode_steps):
   with torch.no_grad():
    dist,_=net(rms.normalize(obs))
    if step%period==0:noise=torch.randn_like(dist.mean)
    action=dist.mean+scale*dist.stddev*noise
   count+=int(alive.sum())*action.shape[-1];clip+=int(((action.abs()>1)&alive[:,None]).sum())
   bounded=action.clamp(-1,1)
   if prev is not None:
    delta=bounded-prev
    if lastdelta is not None:reverse+=int(((delta*lastdelta<0)&alive[:,None]).sum())
    lastdelta=delta
   prev=bounded.clone()
   env.audit_alive=alive.clone();env.audit_step=step
   obs,reward,done,info=env.step(action)
   peak=torch.maximum(peak,env.audit_progress*alive[:,None]);ret+=reward*alive
   passed|=alive&info['episode_success']
   zero+=int(((reward<=0)&alive).sum())
   if step>=args.episode_steps//2:
    latezero+=int(((reward<=0)&alive).sum());latecount+=int(alive.sum())
   alive &= ~done
   if step%240==0:save_json(out/'audit_progress.json',dict(mode=name,step=step,successes=int(passed.sum())))
   if not alive.any():break
  row=dict(speed_events=env.speed_events,mode=name,successes=int(passed.sum()),num_envs=n,success_ids=passed.nonzero().flatten().tolist(),unfinished=int(alive.sum()),clip_fraction=clip/max(count,1),action_reversal_fraction=reverse/max(count,1),late_nonpositive_fraction=latezero/max(latecount,1),peak_mean=dict(zip(env.progress_names,peak.mean(0).tolist())),peak_max=dict(zip(env.progress_names,peak.amax(0).tolist())),reached_half=dict(zip(env.progress_names,(peak>=.5).sum(0).tolist())),reached_full=dict(zip(env.progress_names,(peak>=.999).sum(0).tolist())),returns=ret.tolist(),physical=env.diagnostics())
  results.append(row);save_json(out/'audit_results.json',dict(checkpoint=checkpoint,scene_seed=int(os.environ.get("CONTACT_AUDIT_SCENE_SEED","1062802")),action_seed=1062803,frozen_normalization=True,rows=results))
  print('AUDIT',name,row['successes'],flush=True)
 if state_audit is not None:state_audit.save()
 save_json(out/"completed.json",dict(status="completed",kind="frozen_policy_audit",modes=len(results)))
