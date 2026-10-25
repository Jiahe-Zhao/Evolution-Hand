"""Summarize the bounded experiment without loading the simulator."""
import hashlib,json
from pathlib import Path
import torch
b=Path(__file__).resolve().parent;r=b/'run'
if not (r/'report.json').exists():
 print('Run has no completed report yet');raise SystemExit(1)
d=json.loads((r/'report.json').read_text());policy=json.loads((r/'adaptive_policy.json').read_text())
old=torch.load(r/'inherited_initial.pth',map_location='cpu',weights_only=False)['model']
new=torch.load(r/'ppo_final.pth',map_location='cpu',weights_only=False)['model']
keys=[k for k in old if k.startswith(('a2c_network.actor_mlp.','a2c_network.mu.'))]
changes=torch.cat([(new[k]-old[k]).flatten() for k in keys])
lo=torch.tensor(policy['std_lower']);hi=torch.tensor(policy['std_upper']);std=torch.tensor(policy['std_current']);initial=torch.tensor(policy['std_initial'])
result=dict(before=d['before']['successes'],after=d['after']['successes'],num_envs=d['before']['num_envs'],bc_updates=d['bc_updates'],observation_size=d['observation_size'],
            actor_l2_change=float(changes.norm()),new_local_feature_weight_l2=float(new['a2c_network.actor_mlp.0.weight'][:,108:].norm()),
            std_changed_dimensions=int(((std-initial).abs()>1e-10).sum()),std_within_bounds=bool(((std>=lo-1e-10)&(std<=hi+1e-10)).all()),
            policy=policy,diagnostics=d['after']['diagnostics'],training_diagnostics=json.loads((r/'training_diagnostics.json').read_text()),
            last_complete_evaluation='mean actions, same perturbation set; no independent-seed evaluation',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.glob('*.py')) if not p.name.startswith('._')})
(b/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
