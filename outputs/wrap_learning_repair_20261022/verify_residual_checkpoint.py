"""Verify frozen prior identity and the zero-residual initialization."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--prior',required=True);a=p.parse_args();run=Path(a.run)
source=torch.load(a.prior,map_location='cpu',weights_only=False)['model']
initial=torch.load(run/'bc_initial.pth',map_location='cpu',weights_only=False)
final=torch.load(run/'ppo_final.pth',map_location='cpu',weights_only=False)
assert initial['epoch']==0 and initial['frame']==0
for name in ('a2c_network.mu.weight','a2c_network.mu.bias'):assert torch.count_nonzero(initial['model'][name])==0
prefix='a2c_network.bc_prior.';differences=[]
for key,tensor in final['model'].items():
 if key.startswith(prefix):
  original='a2c_network.'+key[len(prefix):]
  assert torch.equal(tensor,source[original]),key
  differences.append(float((tensor-source[original]).abs().max()))
assert differences
result=dict(initial_epoch=initial['epoch'],initial_frame=initial['frame'],final_epoch=final['epoch'],final_frame=final['frame'],initial_residual_head_max_abs=0.,frozen_prior_max_parameter_change=max(differences),prior_source=a.prior,residual_scale=.02,residual_unbounded=True)
(run/'residual_contract.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
