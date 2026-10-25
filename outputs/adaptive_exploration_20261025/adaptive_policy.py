"""Bounded learned log standard deviation and full-Gaussian update checks."""
import copy
import json
import torch

def gaussian_kl(old_mu,old_std,new_mu,new_std):
    a=old_std.double().clamp_min(1e-12);b=new_std.double().clamp_min(1e-12)
    return (torch.log(b/a)+.5*((a/b).square()+((old_mu.double()-new_mu.double())/b).square()-1)).sum(-1).mean().clamp_min(0)

def install(agent,out,warmup=128,learn_sigma=True,sigma_lr=1e-4,critic_lr=3e-4,actor_lr=1e-7,threshold=.01):
    original=agent.calc_gradients
    actor=[p for n,p in agent.model.named_parameters() if n.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))]
    sigma=agent.model.a2c_network.sigma
    excluded={id(p) for p in actor+[sigma]}
    critic=[p for p in agent.model.parameters() if id(p) not in excluded and p.requires_grad]
    initial_sigma=sigma.detach().clone();lower=initial_sigma+torch.log(torch.tensor(.25,device=sigma.device));upper=initial_sigma+torch.log(torch.tensor(4.,device=sigma.device))
    initial_actor=[p.detach().clone() for p in actor]
    for p in actor:p.requires_grad_(False if warmup else True)
    sigma.requires_grad_(learn_sigma and not warmup)
    if agent.optimizer.state:raise ValueError('Install before optimizer updates')
    agent.optimizer=type(agent.optimizer)([{'params':actor,'lr':actor_lr},{'params':critic,'lr':critic_lr},{'params':[sigma],'lr':sigma_lr}],**agent.optimizer.defaults)
    state=dict(accepted=0,rejected=0,warmup_updates=0,learn_sigma=learn_sigma,actor_lr=actor_lr,sigma_lr=sigma_lr,critic_lr=critic_lr,
               std_initial=initial_sigma.exp().cpu().tolist(),std_lower=lower.exp().cpu().tolist(),std_upper=upper.exp().cpu().tolist(),
               max_accepted_kl=0.,max_proposed_kl=0.,threshold=threshold)
    released=not warmup
    policy=actor+([sigma] if learn_sigma else [])
    def update(batch):
        nonlocal released
        warm=agent.epoch_num<=warmup
        if not warm and not released:
            error=max(float((p-v).abs().max()) for p,v in zip(actor,initial_actor))
            assert error==0 and torch.equal(sigma.detach(),initial_sigma)
            state['warmup_actor_max_change']=error
            for p in actor:p.requires_grad_(True)
            sigma.requires_grad_(learn_sigma);released=True
        for group,rate in zip(agent.optimizer.param_groups,(state['actor_lr'],critic_lr,state['sigma_lr'] if learn_sigma else 0.)):
            group['lr']=rate
        saved=[p.detach().clone() for p in policy]
        optimizer_states={p:copy.deepcopy(agent.optimizer.state[p]) for p in policy if p in agent.optimizer.state}
        original(batch)
        if warm:state['warmup_updates']+=1
        else:
            with torch.no_grad():
                sigma.copy_(sigma.maximum(lower).minimum(upper))
                pred=agent.model({'is_train':False,'prev_actions':None,'obs':batch['obs'],'rnn_states':None})
                kl=float(gaussian_kl(batch['mu'],batch['sigma'],pred['mus'],pred['sigmas']))
            if not torch.isfinite(torch.tensor(kl)) or kl>threshold:
                with torch.no_grad():
                    for p,v in zip(policy,saved):p.copy_(v)
                for p in policy:
                    if p in optimizer_states:agent.optimizer.state[p]=optimizer_states[p]
                    else:agent.optimizer.state.pop(p,None)
                state['rejected']+=1;state['actor_lr']=max(1e-9,state['actor_lr']*.5);state['sigma_lr']=max(1e-7,state['sigma_lr']*.5)
            else:
                state['accepted']+=1;state['max_accepted_kl']=max(state['max_accepted_kl'],kl)
            if torch.isfinite(torch.tensor(kl)):state['max_proposed_kl']=max(state['max_proposed_kl'],kl)
        agent.last_lr=state['actor_lr']
        state['std_current']=sigma.detach().exp().cpu().tolist()
        state['std_max_change']=float((sigma.detach().exp()-initial_sigma.exp()).abs().max())
        (out/'adaptive_policy.json').write_text(json.dumps(state,indent=2))
    agent.calc_gradients=update
    return state
