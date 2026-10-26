"""Final checkpoints: deterministic mean plus three stochastic policy seeds."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from derl_ppo import Policy,Moments,clear_diagnostics,save_json

def run(env,args,out):
 device=env.reset().device;N=env.num_envs;path=Path(args.checkpoint)
 checkpoint=torch.load(path,map_location=device,weights_only=False)
 net=Policy(env.observation_space.shape[0]).to(device);net.load_state_dict(checkpoint['policy']);net.eval()
 rms=Moments((env.observation_space.shape[0],),device)
 for k,v in checkpoint['observation_moments'].items():setattr(rms,k,v)
 initial_mean=rms.mean.clone();initial_var=rms.var.clone();initial_count=rms.count
 rng=np.random.default_rng(1062601);cases=np.column_stack((rng.uniform(-.5,.5,(N,2)),rng.uniform(-1.5,1.5,N)))
 save_json(out/'manifest.json',dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_frames=checkpoint['frames'],source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),observation_seed=1062601,action_seeds=[1062602,1062603,1062604],num_envs=N,episode_steps=1500,mean_trials=1,stochastic_trials=3,training_updates=0,cases=cases.tolist()))
 results=[]
 for mode,seed in [('mean',None)]+[('sample',s) for s in (1062602,1062603,1062604)]:
  label='mean' if seed is None else f'sample_{seed}'
  env.cases=cases.copy();clear_diagnostics(env);env.evaluating=True;obs=env.reset();torch.manual_seed(seed or 1062602);start=time.monotonic();curve=[]
  with torch.no_grad():
   for t in range(1500):
    dist,_=net(rms.normalize(obs));action=dist.mean if mode=='mean' else dist.sample()
    obs,reward,done,info=env.step(action)
    if t%60==59:
     curve.append(dict(seconds=(t+1)/60,held=int(info['held'].sum()),ever_fallen=int((env.episode_falls>0).sum()),holding_3s=int((env.hold>=3).sum())))
  success=(env.hold>=3)&(env.episode_falls==0)
  row=dict(label=label,action_mode=mode,action_seed=seed,observation_seed=1062601,num_envs=N,successes=int(success.sum()),passed=success.cpu().tolist(),ever_held_3s=int((env.best>=3).sum()),unique_fallen=int((env.episode_falls>0).sum()),final_hold=env.hold.cpu().tolist(),diagnostics=env.diagnostics(),curve=curve,duration_s=time.monotonic()-start)
  save_json(out/(label+'.json'),row);results.append(row);save_json(out/'summary.json',results);print('REEVAL',label,row['successes'],'/',N,flush=True)
 assert torch.equal(initial_mean,rms.mean) and torch.equal(initial_var,rms.var) and initial_count==rms.count
 save_json(out/'completed.json',dict(status='completed',trials=len(results),observations_frozen=True,policy_updated=False,checkpoint_frames=checkpoint['frames']))
