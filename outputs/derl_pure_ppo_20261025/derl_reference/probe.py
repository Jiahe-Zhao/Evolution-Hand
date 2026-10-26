"""Execute the unmodified DERL Agent class, isolated from legacy MuJoCo imports."""
import ast,json
from pathlib import Path
import torch
p=Path(__file__).resolve().parent
source=(p/'model.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Agent');ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p/'model.py'),'exec'),ns)
class ConstantAC:
 def __call__(self,obs):return torch.zeros(len(obs),1),torch.distributions.Normal(torch.zeros(len(obs),24),torch.ones(len(obs),24)),None,None
agent=ns['Agent'](ConstantAC());torch.manual_seed(2026);_,a,_=agent.act(torch.zeros(4096,1));assert a.std()>.9
# Illustrative one-joint example, not a measured robot rollout.
torch.manual_seed(2026);mu=.3;sigma=1.;q=.6;span=1.;dt=1/60
sample=torch.randn(200000)*sigma+mu
step=lambda x:(((x.clamp(-1,1)+1)*.5*span)-q).clamp(-.25*dt,.25*dt)
r=dict(source_commit='91e077e355b27bb74fa920c708d606ff9f4d611e',scope='actual upstream Agent class + illustrative control transform; not full DERL reproduction',sampled_action_std=float(a.std()),mean_action_step=float(step(torch.tensor(mu))),sample_action_expected_step=float(step(sample).mean()))
(p/'probe_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
