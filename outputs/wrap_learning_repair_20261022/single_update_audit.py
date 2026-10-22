"""Allow exactly one actor optimizer step after critic-only warmup."""
import json
import torch


def install(agent, warmup_epochs, out, joint_half_range):
    original = agent.calc_gradients
    actor_steps = 0
    skipped = 0
    actor_names = lambda n: n.startswith(('actor_mlp.', 'mu.'))
    prior_initial = {n: p.detach().clone() for n, p in agent.model.a2c_network.bc_prior.named_parameters()}
    def gradients(batch):
        nonlocal actor_steps, skipped
        if agent.epoch_num <= warmup_epochs:
            return original(batch)
        if actor_steps:
            skipped += 1
            return
        before = {n: p.detach().clone() for n, p in agent.model.a2c_network.named_parameters() if actor_names(n)}
        ob = agent._preproc_obs(batch['obs'])
        with torch.no_grad():
            old_mu = agent.model({'is_train': False, 'prev_actions': None, 'obs': ob})['mus'].clone()
        torch.save({k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}, out/'single_update_batch.pt')
        original(batch)
        actor_steps += 1
        with torch.no_grad():
            new_mu = agent.model({'is_train': False, 'prev_actions': None, 'obs': ob})['mus']
            delta = new_mu-old_mu
        prior_change = max(float((p-prior_initial[n]).abs().max()) for n, p in agent.model.a2c_network.bc_prior.named_parameters())
        assert prior_change == 0
        row = dict(actor_optimizer_steps=actor_steps, epoch=agent.epoch_num, batch_size=len(ob),
                   actor_max_parameter_change=max(float((p-before[n]).abs().max()) for n,p in agent.model.a2c_network.named_parameters() if actor_names(n)),
                   frozen_bc_max_parameter_change=prior_change,
                   joint_mean_change_max_rad=float((delta[:,:21]*joint_half_range).abs().max()),
                   wrist_mean_change_max_mm=float(delta[:,21:].abs().max()*250),
                   mean_change_max_in_sigma=float((delta/batch['sigma']).abs().max()),
                   advantages_min=float(batch['advantages'].min()),advantages_max=float(batch['advantages'].max()),
                   action_phase_s=[float(ob[:,66].min()*25),float(ob[:,66].max()*25)],
                   optimizer_lr=[g['lr'] for g in agent.optimizer.param_groups])
        (out/'single_update.json').write_text(json.dumps(row,indent=2))
        # Critic-only checkpoint was saved by the inner warmup wrapper immediately
        # before this step; this checkpoint contains the exactly-once update.
        agent.save(str(out/'single_actor_update'))
    agent.calc_gradients = gradients
    return lambda: dict(actor_optimizer_steps=actor_steps, skipped_actor_minibatches=skipped)
