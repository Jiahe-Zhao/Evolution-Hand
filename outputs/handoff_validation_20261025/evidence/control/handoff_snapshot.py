"""Complete-return value calibration and a local, post-update policy guard."""
import copy
import json
import time
import torch


def calibrate(agent, env, args, out, predict):
    assert not agent.normalize_value, 'Calibration targets use unnormalized reward units'
    agent.model.eval();obs=env.reset();env.evaluating=False
    torch.manual_seed(args.seed+10000)
    observations=[];rewards=[];dones=[]
    actor={k:p.detach().clone() for k,p in agent.model.named_parameters() if k.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))}
    for step in range(args.episode_steps):
        observations.append(obs.detach().clone())
        with torch.no_grad():action=predict(obs)['actions']
        obs,reward,done,_=env.step(action)
        rewards.append(reward.detach().clone()*.01);dones.append(done.detach().clone())
        if step%300==0:print('MC_COLLECT',step,flush=True)
    ob=torch.stack(observations);rew=torch.stack(rewards);done=torch.stack(dones)
    returns=torch.zeros_like(rew);valid=torch.zeros_like(done)
    accumulator=torch.zeros(env.num_envs,device=ob.device);complete=torch.zeros_like(accumulator,dtype=torch.bool)
    for t in range(len(rew)-1,-1,-1):
        accumulator=rew[t]+agent.gamma*(~done[t])*accumulator
        complete=complete|done[t]
        returns[t]=accumulator;valid[t]=complete
    # Split by environment, not adjacent timesteps from the same trajectory.
    heldout=torch.arange(env.num_envs,device=ob.device)%4==0
    x=ob[:,~heldout][valid[:,~heldout]];y=returns[:,~heldout][valid[:,~heldout]]
    vx=ob[:,heldout][valid[:,heldout]];vy=returns[:,heldout][valid[:,heldout]]
    if not len(x) or not len(vx):raise RuntimeError('No complete trajectories available for value calibration')
    net=agent.model.a2c_network
    def value(z):return net.value_act(net.value(net.critic_mlp(z))).squeeze(-1)
    def metrics():
        with torch.no_grad():
            errors=torch.cat([value(vx[i:i+8192])-vy[i:i+8192] for i in range(0,len(vx),8192)])
            return dict(rmse=float(errors.square().mean().sqrt()),bias=float(errors.mean()),target_mean=float(vy.mean()))
    before=metrics();parameters=list(net.critic_mlp.parameters())+list(net.value.parameters())
    optimizer=torch.optim.Adam(parameters,lr=3e-4)
    best=float('inf');best_state=None;history=[]
    for step in range(2000):
        ids=torch.randint(len(x),(4096,),device=x.device)
        loss=(value(x[ids])-y[ids]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        if (step+1)%100==0:
            result=metrics();history.append(dict(step=step+1,**result))
            print('MC_FIT',history[-1],flush=True)
            if result['rmse']<best:
                best=result['rmse'];best_state={k:v.detach().clone() for k,v in net.state_dict().items() if k.startswith(('critic_mlp.','value.'))}
            if result['rmse']<.5 and abs(result['bias'])<.2:break
    net.load_state_dict(best_state,strict=False)
    after=metrics();error=max(float((p-actor[k]).abs().max()) for k,p in agent.model.named_parameters() if k in actor)
    report=dict(before=before,after=after,history=history,training_samples=len(x),validation_samples=len(vx),
                actor_max_change=error,bc_updates=0,accepted=after['rmse']<.5 and abs(after['bias'])<.2,
                split='held-out environments; calibration validation, not independent task evaluation',
                sampled_policy=True,reward_scale=.01,gamma=agent.gamma)
    (out/'critic_calibration.json').write_text(json.dumps(report,indent=2))
    assert error==0.
    agent.save(str(out/'critic_calibrated'))
    if not report['accepted']:raise RuntimeError('Critic calibration did not meet the declared error criterion')


def install_guard(agent,out,threshold=.01):
    """Bound update KL on the sampled batch; not an anchor to the original BC."""
    original=agent.calc_gradients
    actor=[p for n,p in agent.model.named_parameters() if n.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))]
    stats=dict(accepted=0,rejected=0,threshold=threshold,max_accepted_kl=0.,max_proposed_kl=0.)
    def gradients(batch):
        saved=[p.detach().clone() for p in actor]
        original(batch)
        with torch.no_grad():
            result=agent.model({'is_train':False,'prev_actions':None,'obs':batch['obs'],'rnn_states':None})
            kl=float((.5*((result['mus']-batch['mu'])/batch['sigma']).square()).sum(-1).mean())
        stats['max_proposed_kl']=max(stats['max_proposed_kl'],kl)
        if not torch.isfinite(torch.tensor(kl)) or kl>threshold:
            with torch.no_grad():
                for p,v in zip(actor,saved):p.copy_(v)
            for p in actor:agent.optimizer.state.pop(p,None)
            agent.last_lr=max(1e-9,agent.last_lr*.5)
            for group in agent.optimizer.param_groups:group['lr']=agent.last_lr
            stats['rejected']+=1
        else:
            stats['accepted']+=1;stats['max_accepted_kl']=max(stats['max_accepted_kl'],kl)
        stats['learning_rate']=agent.last_lr
        (out/'policy_guard.json').write_text(json.dumps(stats,indent=2))
    agent.calc_gradients=gradients
