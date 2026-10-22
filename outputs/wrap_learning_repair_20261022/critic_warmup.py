"""Fit the separate critic on BC rollouts before allowing actor updates."""
import json

def install(agent,epochs,critic_lr,actor_lr,out):
 actor=[(n,p) for n,p in agent.model.a2c_network.named_parameters() if n.startswith(('actor_mlp.','mu.'))]
 assert actor and all(g.get('weight_decay',0)==0 for g in agent.optimizer.param_groups)
 initial={n:p.detach().clone() for n,p in actor}
 for _,p in actor:p.requires_grad_(False)
 original=agent.calc_gradients;released=False
 def gradients(batch):
  nonlocal released
  warm=agent.epoch_num<=epochs
  if not warm and not released:
   error=max(float((p.detach()-initial[n]).abs().max()) for n,p in actor)
   assert error==0.,'Actor changed during critic-only warmup'
   for _,p in actor:
    p.requires_grad_(True);agent.optimizer.state.pop(p,None)
   agent.last_lr=actor_lr
   (out/'critic_warmup.json').write_text(json.dumps(dict(epochs=epochs,actor_max_parameter_change=error,critic_lr=critic_lr,actor_start_lr=actor_lr,pre_actor_training_diagnostics=agent.vec_env.diagnostics()),indent=2))
   agent.save(str(out/'critic_ready'))
   released=True
  actual_lr=critic_lr if warm else agent.last_lr
  for group in agent.optimizer.param_groups:group['lr']=actual_lr
  agent.writer.add_scalar('diagnostic/critic_only',float(warm),agent.frame)
  agent.writer.add_scalar('diagnostic/effective_optimizer_lr',actual_lr,agent.frame)
  return original(batch)
 agent.calc_gradients=gradients
