"""DERL-style clipped PPO, independently initialized for each morphology.

Reference: agrimgupta92/derl@91e077e355b27bb74fa920c708d606ff9f4d611e.
Task adaptation: true finite-horizon terminals, full vector normalization, GPU batching.
"""
import json,math,time
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
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 torch.manual_seed(getattr(args,'action_seed',args.eval_seed+1))
 original=env.cases.copy();rng=np.random.default_rng(args.eval_seed)
 env.cases=np.column_stack((rng.uniform(-.5,.5,(env.num_envs,2)),rng.uniform(-1.5,1.5,env.num_envs)))
 clear_diagnostics(env);env.evaluating=True;obs=env.reset();net.eval();start=time.monotonic()
 for t in range(32 if args.smoke else args.episode_steps):
  with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.mean if getattr(args,'evaluation_mode','sample')=='mean' else dist.sample()
  obs,_,_,_=env.step(action)
 success=(env.hold>=3)&(env.episode_falls==0)
 row=dict(label=label,full_episode=not args.smoke,successes=int(success.sum()),num_envs=env.num_envs,ever_held_3s=int((env.best>=3).sum()),passed=success.cpu().tolist(),
          final_hold_s=env.hold.cpu().tolist(),cases=env.cases.tolist(),eval_seed=args.eval_seed,action_mode=getattr(args,'evaluation_mode','sample'),action_seed=getattr(args,'action_seed',args.eval_seed+1),goal_contract=env.goals.contract(),goal_heights_m=env.goals.target.cpu().tolist(),diagnostics=env.diagnostics(),duration_s=time.monotonic()-start)
 row['by_goal']=[dict(height_m=float(h),environments=int((env.goals.target==h).sum()),successes=int((success & (env.goals.target==h)).sum())) for h in env.goals.levels]
 save_json(out/(label+'.json'),row);print('EVAL',label,row['successes'],'/',env.num_envs,flush=True)
 env.cases=original;env.evaluating=False;clear_diagnostics(env);net.train()
 torch.set_rng_state(cpu_rng)
 if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
 return row

def run(env,args,out):
 torch.manual_seed(args.seed);np.random.seed(args.seed)
 device=env.reset().device;net=Policy(env.observation_space.shape[0]).to(device)
 optimizer=torch.optim.Adam(net.parameters(),lr=3e-4,eps=1e-5)
 rms=Moments((env.observation_space.shape[0],),device);rrms=Moments((),device)
 N,T=env.num_envs,args.horizon;iterations=args.steps//(N*T)
 config=dict(reference_commit='91e077e355b27bb74fa920c708d606ff9f4d611e',network=[64,64],activation='tanh',separate_actor_critic=True,
             initial_lr=3e-4,lr_schedule='linear',gamma=.99,gae_lambda=.95,clip=.2,optimizer_epochs=4,minibatch=512,value_coefficient=.5,
             entropy_coefficient=.01,max_grad_norm=.5,adam_eps=1e-5,kl_early_stop=.2,initial_std=1.,learn_std=True,
             rollout=T,num_envs=N,batch=N*T,iterations=iterations,requested_steps=args.steps,actual_steps=iterations*N*T,
             observation_size=env.observation_space.shape[0],observation_normalization=True,reward_return_std_normalization=True,
             bc_updates=0,parent_policy_loaded=False,critic_warmup=0,episode_steps=args.episode_steps,
             reward='dense approach + opposition + goal-conditioned lift + holding within goal tolerance + terminal success; unchanged physical costs',
             differences_from_derl=['Isaac/PhysX contact task, not MuJoCo locomotion','GPU environment batch size','all 152 observation values normalized','25s endpoint is a true terminal, not a time-limit truncation','absolute position targets with existing slew limits'])
 config.update(goal_contract=env.goals.contract(),observation_contract='152D goal-conditioned; incompatible with legacy 148D checkpoints')
 save_json(out/'effective_config.json',config)
 def save(name,frames):
  torch.save(dict(policy=net.state_dict(),optimizer=optimizer.state_dict(),observation_moments=rms.state(),reward_moments=rrms.state(),frames=frames,config=config),out/(name+'.pth'))
 save('random_initial',0)
 # No learned action or demonstration is used for initialization validation.
 obs=env.reset();hold=torch.cat((obs[:,:21],torch.zeros(N,3,device=device)),1)
 for _ in range(60):obs,_,_,_=env.step(hold)
 physical=env.diagnostics();save_json(out/'initial_physics.json',physical)
 if physical['max_digit_force_N']>25 or physical['max_limit_rad']>.05:
  raise RuntimeError('Initial physical settling invalid; distinguish this from PPO task failure')
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
   discounted=.99*discounted+reward;rrms.update(discounted)
   norm_reward=(reward/(rrms.var+1e-8).sqrt()).float().clamp(-10,10);discounted[done]=0
   rms.update(obs)
   for key,val in zip(storage,(normalized,action,logp,value,norm_reward,done)):storage[key].append(val.detach().clone())
   window_episodes+=int(done.sum());window_success+=int(info['episode_success'].sum());raw_return+=float(reward.mean());held_samples+=int(info['held'].sum())
  with torch.no_grad():_,tail=net(rms.normalize(obs))
  data={k:torch.stack(v) for k,v in storage.items()}
  ret,adv=gae(data['rew'],data['value'],data['done'],tail)
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
             training_episodes=train_episodes,training_successes=train_success,bc_updates=0,checkpoint_loaded=False,
             total_training_and_eval_s=time.monotonic()-run_started,source_sha256=__import__('hashlib').sha256(Path(args.source).read_bytes()).hexdigest())
 save_json(out/'report.json',report);print('FINISHED',before['successes'],evaluations[-1]['successes'],flush=True)

 if not args.smoke:
  final_trials=[]
  for mode,seed in [('mean',args.eval_seed+1)]+[('sample',args.eval_seed+i) for i in (1,2,3)]:
   args.evaluation_mode=mode;args.action_seed=seed
   final_trials.append(evaluate(env,net,rms,args,out,f'final_{mode}_{seed}'))
  save_json(out/'final_trials.json',final_trials)
  save_json(out/'completed.json',dict(status='completed',frames=frames,full_episode=True,trials=len(final_trials)))
