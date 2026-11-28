"""Finite short-PPO integration check using versioned task signatures."""
import json,math,time
import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self,obs,actions):
        super().__init__()
        def network(output):
            layers=[nn.Linear(obs,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,output)]
            for layer in layers:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight);nn.init.zeros_(layer.bias)
            return nn.Sequential(*layers)
        self.actor=network(actions);self.critic=network(1);self.log_std=nn.Parameter(torch.zeros(actions))
    def forward(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.exp()),self.critic(x).squeeze(-1)

class Moments:
    def __init__(self,shape,device):
        self.mean=torch.zeros(shape,device=device,dtype=torch.float64);self.var=torch.ones_like(self.mean);self.count=1e-4
    def update(self,x):
        x=x.double();n=len(x);delta=x.mean(0)-self.mean;total=self.count+n
        self.var=(self.var*self.count+x.var(0,unbiased=False)*n+delta.square()*self.count*n/total)/total
        self.mean+=delta*n/total;self.count=total
    def normalize(self,x):return ((x-self.mean)/(self.var+1e-8).sqrt()).float().clamp(-10,10)
    def state(self):return dict(mean=self.mean,var=self.var,count=self.count)

def run(env,out,iterations=4):
    c=env.config['ppo'];n=env.num_envs;t=c['rollout_steps'];device=env.device
    if n*t<c['minibatch']:raise ValueError('Rollout smaller than minibatch')
    torch.manual_seed(env.seed);net=Policy(env.observation_size,env.action_size).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=c['learning_rate'],eps=c['adam_epsilon'])
    rms=Moments((env.observation_size,),device);rrms=Moments((),device)
    obs=env.reset();rms.update(obs);discounted=torch.zeros(n,device=device);rows=[];start=time.monotonic()
    episode_returns=torch.zeros(n,device=device)
    def save(name,iteration):
        target=out/name;temporary=out/(name+'.tmp')
        torch.save(dict(policy=net.state_dict(),optimizer=optimizer.state_dict(),observation_moments=rms.state(),reward_moments=rrms.state(),interface_signature=env.signature,config=env.config,iteration=iteration,transitions=iteration*n*t),temporary)
        temporary.replace(target)
    for iteration in range(iterations):
        lr=c['learning_rate']*(1-iteration/iterations)
        for group in optimizer.param_groups:group['lr']=lr
        data={k:[] for k in ['obs','action','logp','value','reward','done']};successes=0;episodes=0;completed_returns=[]
        for _ in range(t):
            normalized=rms.normalize(obs)
            with torch.no_grad():dist,value=net(normalized);action=dist.sample();logp=dist.log_prob(action).sum(-1)
            obs,reward,term,trunc,info=env.step(action);done=term|trunc
            episode_returns+=reward
            episodes+=int(done.sum());completed_returns.extend(episode_returns[done].detach().cpu().tolist());episode_returns[done]=0
            # Time limits bootstrap terminal state; true task terminations do not.
            with torch.no_grad():_,terminal_value=net(rms.normalize(info['terminal_observation']))
            discounted=c['gamma']*discounted+reward;rrms.update(discounted)
            normalized_reward=(reward/(rrms.var+1e-8).sqrt()).float().clamp(-10,10)
            normalized_reward+=c['gamma']*terminal_value*(trunc&~term).float()
            discounted[done]=0;rms.update(obs)
            for key,value in zip(data,[normalized,action,logp,value,normalized_reward,done]):data[key].append(value.detach().clone())
            successes+=int(info['episode_success'].sum())
        data={k:torch.stack(v) for k,v in data.items()}
        with torch.no_grad():_,tail=net(rms.normalize(obs))
        advantage=torch.zeros_like(data['reward']);running=torch.zeros(n,device=device)
        for step in reversed(range(t)):
            following=tail if step==t-1 else data['value'][step+1];live=(~data['done'][step]).float()
            delta=data['reward'][step]+c['gamma']*following*live-data['value'][step]
            running=delta+c['gamma']*c['gae_lambda']*live*running;advantage[step]=running
        returns=advantage+data['value'];advantage=(advantage-advantage.mean())/(advantage.std()+1e-5)
        flat={k:v.flatten(0,1) for k,v in data.items()};flat['returns']=returns.flatten();flat['advantage']=advantage.flatten()
        updates=0;stop=False;losses=[]
        for _ in range(c['epochs']):
            for ids in torch.randperm(n*t,device=device).split(c['minibatch']):
                if len(ids)<c['minibatch']:continue
                dist,value=net(flat['obs'][ids]);logp=dist.log_prob(flat['action'][ids]).sum(-1)
                kl=(flat['logp'][ids]-logp).mean()
                if float(kl.detach())>c['kl_early_stop']:stop=True;break
                ratio=(logp-flat['logp'][ids]).exp();adv=flat['advantage'][ids]
                actor=-torch.minimum(ratio*adv,ratio.clamp(1-c['clip'],1+c['clip'])*adv).mean()
                clipped=flat['value'][ids]+(value-flat['value'][ids]).clamp(-c['value_clip'],c['value_clip'])
                critic=.5*torch.maximum((value-flat['returns'][ids]).square(),(clipped-flat['returns'][ids]).square()).mean()
                loss=actor+c['value_coefficient']*critic-c['entropy_coefficient']*dist.entropy().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),c['max_grad_norm']);optimizer.step()
                updates+=1;losses.append(float(loss.detach()))
            if stop:break
        rows.append(dict(iteration=iteration+1,transitions=(iteration+1)*n*t,updates=updates,loss_mean=sum(losses)/max(1,len(losses)),successes=successes,episodes=episodes,episode_return_mean=sum(completed_returns)/max(1,len(completed_returns)),elapsed_s=time.monotonic()-start,transitions_per_second=(iteration+1)*n*t/(time.monotonic()-start),std=net.log_std.exp().detach().tolist()))
        (out/'ppo_progress.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(rows[-1]),flush=True)
        if (iteration+1)%32==0 or iteration+1==iterations:
            save('latest.pth',iteration+1)
            save(f'checkpoint_{(iteration+1)*n*t:09d}.pth',iteration+1)
    assert sum(row['updates'] for row in rows)>0
    torch.save(dict(policy=net.state_dict(),optimizer=optimizer.state_dict(),observation_moments=rms.state(),reward_moments=rrms.state(),interface_signature=env.signature,config=env.config),out/'short_ppo.pth')
    # Checkpoint is tied to this exact observation/action/scene interface.
    restored=torch.load(out/'short_ppo.pth',map_location=device,weights_only=False);assert restored['interface_signature']==env.signature
    net.load_state_dict(restored['policy'])
    report=dict(passed=True,transitions=iterations*n*t,elapsed_s=time.monotonic()-start,updates=sum(r['updates'] for r in rows),checkpoint_roundtrip=True,scope='PPO integration and finite updates; not convergence',iterations=rows)
    (out/'short_ppo.json').write_text(json.dumps(report,indent=2));return report
