"""Frozen BC controller plus an initially zero, unbounded learned residual.

The prior is embedded in the checkpoint. Original physical limits still apply;
no new task gate, attachment, or hard bound is introduced on the residual.
"""
import copy
import torch

def install(network,scale=.02):
 prior=copy.deepcopy(network).eval()
 for parameter in prior.parameters():parameter.requires_grad_(False)
 network.add_module('bc_prior',prior)
 with torch.no_grad():network.mu.weight.zero_();network.mu.bias.zero_()
 def combine(module,inputs,output):
  mu,sigma,value,states=output
  with torch.no_grad():base=module.bc_prior(inputs[0])[0]
  return base+scale*mu,sigma,value,states
 network.register_forward_hook(combine)
