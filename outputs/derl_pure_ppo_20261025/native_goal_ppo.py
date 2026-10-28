"""DERL-style clipped PPO, independently initialized for each morphology.

Reference: agrimgupta92/derl@91e077e355b27bb74fa920c708d606ff9f4d611e.
Task adaptation: true finite-horizon terminals, full vector normalization, GPU batching.
"""
import json,math,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Moments:
 def __init__(self,shape,device):
  self.mean=torch.zeros(shape,device=device,dtype=torch.float64);self.var=torch.ones_like(self.mean);self.count=1e-4
 def update(self,x):
  x=x.double();n=x.shape[0];mean=x.mean(0);var=x.var(0,unbiased=False);delta=mean-self.mean;total=self.count+n
  self.var=(self.var*self.count+var*n+delta.square()*self.count*n/total)/total
  self.mean+=delta*n/total;self.count=total
 def normalize(self,x):return ((x-self.mean)/(self.var+1e-8).sqrt()).float().clamp(-10,10)
 def state(self):return dict(mean=self.mean,var=self.var,count=self.count)

def layer(a,b):
 l=nn.Linear(a,b);nn.init.orthogonal_(l.weight);nn.init.zeros_(l.bias);return l
class Policy(nn.Module):
 def __init__(self,obs_size,act_size=24):
  super().__init__()
  self.actor=nn.Sequential(layer(obs_size,64),nn.Tanh(),layer(64,64),nn.Tanh(),layer(64,act_size))
  self.critic=nn.Sequential(layer(obs_size,64),nn.Tanh(),layer(64,64),nn.Tanh(),layer(64,1))
  self.log_std=nn.Parameter(torch.zeros(act_size))
 def forward(self,obs):
  mu=self.actor(obs);dist=torch.distributions.Normal(mu,self.log_std.exp().expand_as(mu))
  return dist,self.critic(obs).squeeze(-1)

def gae(rewards,values,dones,tail,gamma=.99,lam=.95):
 advantage=torch.zeros_like(rewards);running=torch.zeros_like(tail)
 for t in reversed(range(len(rewards))):
  next_value=tail if t==len(rewards)-1 else values[t+1]
  live=(~dones[t]).float();delta=rewards[t]+gamma*next_value*live-values[t]
  running=delta+gamma*lam*live*running;advantage[t]=running
 return advantage+values,advantage

def clipped_losses(value,logp,entropy,batch):
 ratio=(logp-batch['logp']).exp();adv=batch['adv']
 actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
 vclip=batch['value']+(value-batch['value']).clamp(-.2,.2)
 critic=.5*torch.maximum((value-batch['ret']).square(),(vclip-batch['ret']).square()).mean()
 return actor+.5*critic-.01*entropy.mean(),actor,critic

def save_json(path,data):
 temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def clear_diagnostics(env):
 env.force_exceedances=0;env.max_speed=0.;env.max_force=0.;env.max_limit=0.;env.falls=0;env.episodes=0;env.anomalies=[];env.successful_terminal_episodes=0;env.max_terminal_shaping_residual=0.

def evaluate(env,net,rms,args,out,label):
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 env.reset_seed=args.eval_seed;obs=env.reset();net.eval()
 # Environment reset seeds Torch too. Set the action stream afterwards so
 # different action_seed trials really sample different trajectories.
 torch.manual_seed(getattr(args,'action_seed',args.eval_seed+1))
 alive=torch.ones(env.num_envs,dtype=torch.bool,device=obs.device)
 success=torch.zeros_like(alive);lengths=torch.zeros_like(alive,dtype=torch.long)
 returns=torch.zeros(env.num_envs,device=obs.device)
 for t in range(min(32,args.episode_steps) if getattr(args,'smoke',False) else args.episode_steps):
  with torch.no_grad():
   dist,_=net(rms.normalize(obs));action=dist.mean if getattr(args,'evaluation_mode','sample')=='mean' else dist.sample()
  obs,reward,done,info=env.step(action)
  success |= alive & info['episode_success'];lengths+=alive.long();returns+=reward*alive
  alive &= ~done
  if not alive.any():break
 row=dict(label=label,full_episode=not getattr(args,'smoke',False),successes=int(success.sum()),num_envs=env.num_envs,passed=success.cpu().tolist(),
  first_episode_lengths=lengths.cpu().tolist(),unfinished_at_cap=int(alive.sum()),returns=returns.cpu().tolist(),
  action_mode=getattr(args,'evaluation_mode','sample'),action_seed=getattr(args,'action_seed',args.eval_seed+1),eval_seed=args.eval_seed,
  goal_contract=env.contract())
 if env.task=='strike' and hasattr(env,'phase'):
  row['per_goal']={str(k):{'successes':int(success[env.phase==k].sum()),'episodes':int((env.phase==k).sum())} for k in (1,2)}
 save_json(out/(label+'.json'),row);print('EVAL',label,row['successes'],'/',env.num_envs,flush=True)
 env.reset_seed=None;net.train();torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
 return row

def run(env,args,out):
 if os.environ.get('FORAGE_POLICY_VIDEO'):
  from forage_policy_video import record
  return record(env,args,out)
 if os.environ.get("CONTACT_POLICY_AUDIT"):
  from contact_policy_audit import run as audit
  return audit(env,args,out)
 import shutil,hashlib
 source_files=['native_goal_ppo.py','native_goal_adapter.py','native_goal_worker.py','native_contact_hooks.py']
 if hasattr(env.raw,'whole_hand_tactile'):source_files.append('whole_hand_tactile.py')
 if getattr(args,'forage_contact_repair',False):source_files.append('forage_contact_repair.py')
 if os.environ.get('CONTACT_CURRICULUM')=='1':source_files.append('contact_curriculum.py')
 if os.environ.get('BRANCH_PLACEMENT_MANIFEST'):source_files.append('branch_local_scene.py')
 if os.environ.get('BRANCH_COMPARISON_HAND'):source_files.extend(['branch_comparison.py','branch_comparison_geometry.py','branch_comparison_protocol.py'])
 if os.environ.get('CONTACT_PROGRESS_REWARD')=='1':source_files.append('contact_progress_reward.py')
 if os.environ.get('FORAGE_STATE_GUARD')=='1':source_files.append('forage_state_audit.py')
 if os.environ.get('STRIKE_PROGRESS_REWARD')=='1':source_files.extend(['strike_progress_reward.py','contact_progress_reward.py','contact_curriculum.py'])
 snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
 hashes={}
 for name in source_files:
  source=Path(__file__).with_name(name);shutil.copy2(source,snapshot/name);hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
 save_json(out/'source_hashes.json',hashes)
 torch.manual_seed(args.seed);np.random.seed(args.seed)
 device=env.reset().device;net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(device)
 optimizer=torch.optim.Adam(net.parameters(),lr=3e-4,eps=1e-5)
 rms=Moments((env.observation_space.shape[0],),device);rrms=Moments((),device)
 parent=os.environ.get('PPO_PARENT_CHECKPOINT')
 if parent:
  checkpoint=torch.load(parent,map_location=device,weights_only=False)
  net.load_state_dict(checkpoint['policy'])
  for moments,key in ((rms,'observation_moments'),(rrms,'reward_moments')):
   state=checkpoint[key];moments.mean=state['mean'].to(device);moments.var=state['var'].to(device);moments.count=state['count']
 N,T=env.num_envs,args.horizon;iterations=args.steps//(N*T)
 gamma=float(os.environ.get('CONTACT_PPO_GAMMA','.99'))
 if not 0<gamma<1:raise ValueError('Invalid discount factor')
 config=dict(reference_commit='91e077e355b27bb74fa920c708d606ff9f4d611e',network=[64,64],activation='tanh',separate_actor_critic=True,
             initial_lr=3e-4,lr_schedule='linear',gamma=.99,gae_lambda=.95,clip=.2,optimizer_epochs=4,minibatch=512,value_coefficient=.5,
             entropy_coefficient=.01,max_grad_norm=.5,adam_eps=1e-5,kl_early_stop=.2,initial_std=1.,learn_std=True,
             rollout=T,num_envs=N,batch=N*T,iterations=iterations,requested_steps=args.steps,actual_steps=iterations*N*T,
             observation_size=env.observation_space.shape[0],observation_normalization=True,reward_return_std_normalization=True,
             bc_updates=0,parent_policy_loaded=bool(parent),critic_warmup=0,episode_steps=args.episode_steps,
             reward='potential difference plus native final success; no grasp milestone reward',
             differences_from_derl=['Isaac/PhysX contact task, not MuJoCo locomotion','GPU environment batch size','native task joint graph observations and goal normalized','finite-horizon terminal evaluation','native joint graph action bounds and slew limits'])
 config.update(goal_contract=env.goals.contract(),observation_contract='native joint graph plus task goal; fresh per-task policy')
 config.update(parent_checkpoint=parent,goal_contract=env.contract(),reward='potential difference plus current curriculum success' if os.environ.get('CONTACT_CURRICULUM')=='1' else config['reward'])
 config['gamma']=gamma
 if os.environ.get('CONTACT_PROGRESS_REWARD')=='1':config['reward']='bounded per-episode progress increments plus unchanged native success; soft physical/time costs'
 save_json(out/'effective_config.json',config)
 def save(name,frames):
  torch.save(dict(policy=net.state_dict(),optimizer=optimizer.state_dict(),observation_moments=rms.state(),reward_moments=rrms.state(),frames=frames,config=config),out/(name+'.pth'))
 save('random_initial',0)
 obs=env.reset();rms.update(obs)
 before=evaluate(env,net,rms,args,out,'before_training')
 obs=env.reset();rms.update(obs);discounted=torch.zeros(N,device=device)
 train_episodes=train_success=0;next_eval=args.eval_every;best=-1;evaluations=[dict(frames=0,**before)]
 started=time.monotonic();run_started=started
 for iteration in range(iterations):
  lr=3e-4*(1-iteration/iterations)
  for group in optimizer.param_groups:group['lr']=lr
  storage={k:[] for k in ('obs','act','logp','value','rew','done')};window_success=window_episodes=0;raw_return=0.;held_samples=0
  for step in range(T):
   normalized=rms.normalize(obs)
   with torch.no_grad():dist,value=net(normalized);action=dist.sample();logp=dist.log_prob(action).sum(-1)
   if not torch.isfinite(action).all():raise RuntimeError('Nonfinite policy action')
   obs,reward,done,info=env.step(action)
   discounted=gamma*discounted+reward;rrms.update(discounted)
   norm_reward=(reward/(rrms.var+1e-8).sqrt()).float().clamp(-10,10);discounted[done]=0
   rms.update(obs)
   for key,val in zip(storage,(normalized,action,logp,value,norm_reward,done)):storage[key].append(val.detach().clone())
   window_episodes+=int(done.sum());window_success+=int(info['episode_success'].sum());raw_return+=float(reward.mean());held_samples+=int(info['held'].sum())
  with torch.no_grad():_,tail=net(rms.normalize(obs))
  data={k:torch.stack(v) for k,v in storage.items()}
  ret,adv=gae(data['rew'],data['value'],data['done'],tail,gamma=gamma)
  adv=(adv-adv.mean())/(adv.std()+1e-5)
  flat=dict(obs=data['obs'].flatten(0,1),act=data['act'].flatten(0,1),logp=data['logp'].flatten(),value=data['value'].flatten(),ret=ret.flatten(),adv=adv.flatten())
  stop=False;updates=0;kls=[];losses=[]
  for epoch in range(4):
   for idx in torch.randperm(N*T,device=device).split(512):
    if len(idx)<512:continue
    batch={k:v[idx] for k,v in flat.items()};dist,val=net(batch['obs']);logp=dist.log_prob(batch['act']).sum(-1)
    kl=float((batch['logp']-logp).mean().detach());kls.append(kl)
    if kl>.2:stop=True;break
    loss,al,vl=clipped_losses(val,logp,dist.entropy(),batch)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step();updates+=1
    losses.append([float(al.detach()),float(vl.detach())])
   if stop:break
  frames=(iteration+1)*N*T;train_episodes+=window_episodes;train_success+=window_success
  row=dict(iteration=iteration+1,frames=frames,lr=lr,elapsed_s=time.monotonic()-started,training_fps=frames/(time.monotonic()-started),
           completed_episodes=train_episodes,successful_episodes=train_success,new_episodes=window_episodes,new_successes=window_success,
           mean_raw_reward=raw_return/T,held_fraction=held_samples/(N*T),reward_components_last=env.reward_terms,peak_lift_m=float(env.peak_lift.max()),optimizer_updates=updates,kl_early_stop=stop,max_approx_kl=max(kls) if kls else None,
           std_min=float(net.log_std.exp().min().detach()),std_max=float(net.log_std.exp().max().detach()),physical=env.diagnostics())
  with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save_json(out/'progress.json',row)
  print('PPO',frames,'/',iterations*N*T,'success',window_success,'/',window_episodes,'fps',round(row['training_fps']),flush=True)
  if frames>=next_eval or iteration==iterations-1:
   save('latest',frames);eval_start=time.monotonic()
   evaluation=evaluate(env,net,rms,args,out,f'eval_{frames}')
   evaluations.append(dict(frames=frames,**evaluation));save_json(out/'evaluations.json',evaluations)
   if evaluation['successes']>best:best=evaluation['successes'];save('best',frames)
   next_eval+=args.eval_every;obs=env.reset();rms.update(obs);discounted.zero_()
   started+=time.monotonic()-eval_start
 save('final',frames)
 report=dict(status='completed',config=config,before=before,after=evaluations[-1],best_successes=best,evaluations=evaluations,
             training_episodes=train_episodes,training_successes=train_success,bc_updates=0,checkpoint_loaded=bool(parent),
             total_training_and_eval_s=time.monotonic()-run_started,source_sha256=__import__('hashlib').sha256(Path(args.source).read_bytes()).hexdigest())
 save_json(out/'report.json',report);print('FINISHED',before['successes'],evaluations[-1]['successes'],flush=True)

 if not args.smoke:
  final_trials=[evaluations[-1]]
  for mode,seed in [('sample',args.eval_seed+2),('mean',args.eval_seed+1)]:
   args.evaluation_mode=mode;args.action_seed=seed
   final_trials.append(evaluate(env,net,rms,args,out,f'final_{mode}_{seed}'))
  save_json(out/'final_trials.json',final_trials)
  save_json(out/'completed.json',dict(status='completed',frames=frames,full_episode=True,trials=len(final_trials)))
