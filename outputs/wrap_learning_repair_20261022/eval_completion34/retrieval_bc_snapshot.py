"""Local supervised BC baseline; no online teacher or simulator-state writes.

The retained demonstration states/actions are checkpoint buffers. PPO learns an
unconstrained residual around the retrieved absolute position command.
"""
import torch

def install(network,data,lookahead=0):
 obs=data['obs'].float();actions=data['actions'].float()
 if lookahead:
  chunks=[];start=0
  for source in data.get('sources',[dict(samples=len(obs))]):
   count=source['samples'];part_obs=obs[start:start+count];part_actions=actions[start:start+count]
   ticks=torch.unique(part_obs[:,66]);T=len(ticks);n=count//T
   assert T*n==count and torch.allclose(part_obs[:,66].reshape(T,n)[:,0],ticks)
   forward=(torch.arange(T,device=obs.device)+lookahead).clamp_max(T-1)
   chunks.append(part_actions.reshape(T,n,24)[forward].reshape(-1,24));start+=count
  actions=torch.cat(chunks)
 ids=list(range(55))+list(range(61,73))
 scale=torch.tensor([5.]*21+[20.]*21+[25.]*6+[5.]*3+[.2]*4+[.05]*5+[.05]+[.05]*5+[.2],device=obs.device)
 center=obs[0,ids].clone();features=(obs[:,ids]-center)*scale
 network.register_buffer('bc_features',features)
 network.register_buffer('bc_actions',actions)
 network.register_buffer('bc_center',center)
 network.register_buffer('bc_scale',scale)
 with torch.no_grad():network.mu.weight.zero_();network.mu.bias.zero_()
 def hook(module,inputs,output):
  mu,sigma,value,states=output;ob=inputs[0]['obs']
  with torch.no_grad():
   queries=(ob[:,ids]-module.bc_center)*module.bc_scale
   commands=[]
   for q in queries.split(128):
    distance=torch.cdist(q,module.bc_features,compute_mode='donot_use_mm_for_euclid_dist')
    ix=distance.argmin(1);commands.append(module.bc_actions[ix])
   reference=torch.cat(commands)
  return reference+.02*mu,sigma,value,states
 network.register_forward_hook(hook)
