"""Measure actual actor adaptation in completed arms; no simulator required."""
import json
from pathlib import Path
import torch
base=Path(__file__).resolve().parent
result={}
for name in ('standard','calibrated'):
    folder=base/name
    if not (folder/'report.json').exists():
        continue
    initial=torch.load(folder/'inherited_initial.pth',map_location='cpu',weights_only=False)['model']
    final=torch.load(folder/'ppo_final.pth',map_location='cpu',weights_only=False)['model']
    keys=[k for k in final if k.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))]
    difference=torch.cat([(final[k]-initial[k]).reshape(-1) for k in keys])
    new=final['a2c_network.actor_mlp.0.weight'][:,67:]
    result[name]=dict(actor_max_absolute_change=float(difference.abs().max()),
                      actor_l2_change=float(difference.norm()),
                      appended_observation_weight_l2=float(new.norm()),
                      appended_observation_nonzero=int(torch.count_nonzero(new)))
(base/'actor_changes.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
