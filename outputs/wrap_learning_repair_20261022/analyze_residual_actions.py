"""Quantify learned corrections on recorded physical states, not a rollout proof."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
s=torch.load(root/'ppo_residual25/ppo_final.pth',map_location='cpu',weights_only=False)['model']
d=torch.load(root/'recovery_teacher03/demonstrations.pt',map_location='cpu',weights_only=False)['obs'][:,:67]
def residual(ob):
 x=ob
 for i in (0,2,4,6):x=F.elu(F.linear(x,s[f'a2c_network.actor_mlp.{i}.weight'],s[f'a2c_network.actor_mlp.{i}.bias']))
 return .02*F.linear(x,s['a2c_network.mu.weight'],s['a2c_network.mu.bias'])
with torch.no_grad():delta=residual(d)
source=Path('/home/zjh/Evolution_PC/outputs/motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf')
lim={j.attrib['name']:float(j.find('limit').attrib['upper'])-float(j.find('limit').attrib['lower']) for j in ET.parse(source).getroot().findall('joint') if j.attrib['type']!='fixed' and j.find('limit') is not None}
audit=json.loads((root.parent/'contact_control_calibration_20261021/wrap04/audit.json').read_text())
names=[n for n in audit['joint_names'] if n in lim];assert len(names)==21
angular=delta[:,:21]*torch.tensor([lim[n]/2 for n in names]);wrist=delta[:,21:]*250
T=len(torch.unique(d[:,66]));n=len(d)//T
report=dict(scope='Corrections on stored teacher states; does not establish the size along the policy rollout.',joint_rmse_rad=float(angular.square().mean().sqrt()),joint_max_abs_rad=float(angular.abs().max()),wrist_rmse_mm=float(wrist.square().mean().sqrt()),wrist_max_abs_mm=float(wrist.abs().max()),nominal_samples=[dict(t_s=t/60,wrist_delta_mm=wrist[t*n].tolist(),max_joint_delta_rad=float(angular[t*n].abs().max())) for t in (0,45,300,600,900,1200)])
(root/'residual_constant_bias.json').write_text(json.dumps(delta.mean(0).tolist(),indent=2))
(root/'ppo_residual25/residual_action_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
