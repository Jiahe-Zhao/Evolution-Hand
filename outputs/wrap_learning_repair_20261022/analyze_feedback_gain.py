"""Partial target-memory Jacobian, not the full hand/physics stability Jacobian."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='bc_position01/bc_initial.pth');p.add_argument('--data',default='recovery_teacher03/demonstrations.pt');p.add_argument('--output',default='feedback_gain.json');a=p.parse_args()
torch.set_num_threads(1)
s=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'];d=torch.load(a.data,map_location='cpu',weights_only=False)
D=s['a2c_network.actor_mlp.0.weight'].shape[1]
def predict(ob):
 x=ob
 for i in (0,2,4,6):x=F.elu(F.linear(x,s[f'a2c_network.actor_mlp.{i}.weight'],s[f'a2c_network.actor_mlp.{i}.bias']))
 return F.linear(x,s['a2c_network.mu.weight'],s['a2c_network.mu.bias'])
ids=list(range(21,42))+list(range(45,48));scale=torch.tensor([1.]*21+[5.]*3)
# Data is time-major, successful environment-major within each time step.
T=len(torch.unique(d['obs'][:,66]));n=len(d['obs'])//T;rows=[]
for t in (0,45,300,600,900,1200):
 x=d['obs'][t*n,:D].clone().requires_grad_(True)
 jac=torch.autograd.functional.jacobian(predict,x)
 A=scale[:,None]*jac[:,ids]
 B=scale[:,None]*jac[:,list(range(21))+list(range(42,45))]
 rows.append(dict(t_s=t/60,spectral_radius=float(torch.linalg.eigvals(A).abs().max()),largest_singular_value=float(torch.linalg.svdvals(A)[0]),instant_tracking_approx_spectral_radius=float(torch.linalg.eigvals(A+B).abs().max())))
result=dict(checkpoint=a.checkpoint,scope='Partial previous-target feedback with all other observations held fixed; excludes plant dynamics and therefore does not prove complete closed-loop stability or instability.',samples=rows)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
