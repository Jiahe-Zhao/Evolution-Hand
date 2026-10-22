"""Soft policy-mean KL penalty to the starting BC on PPO rollout states."""
import copy
import torch

def install(agent,coefficient):
 reference=copy.deepcopy(agent.model.a2c_network).eval()
 for parameter in reference.parameters():parameter.requires_grad_(False)
 cache={};original=agent.actor_loss_func
 def observe(module,inputs,result):
  feed=inputs[0]
  if not feed.get('is_train',False):return
  with torch.no_grad():
   prior=reference({'obs':feed['obs'],'rnn_states':None})[0]
  # Sigmas are fixed in this experiment. This is the exact equal-variance KL.
  cache['penalty']=.5*((result['mus']-prior)/result['sigmas'].detach()).square().sum(-1)
 def actor_loss(*args,**kwargs):
  base=original(*args,**kwargs);penalty=cache['penalty']
  agent.writer.add_scalar('diagnostic/bc_anchor_kl',float(penalty.detach().mean()),agent.frame)
  return base+coefficient*penalty
 agent.model.register_forward_hook(observe);agent.actor_loss_func=actor_loss
 return reference
