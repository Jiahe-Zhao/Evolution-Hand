"""Paired rollout and same-state action audit; no claims of full causal isolation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run).resolve();root=run.parent
torch.set_num_threads(4)
b=np.load(run/'before_ppo_trace.npz');c=np.load(run/'after_ppo_trace.npz')
cfg=yaml.safe_load((run/'effective_agent.yaml').read_text())['params']['config']
source=Path('/home/zjh/Evolution_PC/outputs/motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf')
limits={j.attrib['name']:(float(j.find('limit').attrib['upper'])-float(j.find('limit').attrib['lower']))/2 for j in ET.parse(source).getroot().findall('joint') if j.attrib['type']!='fixed' and j.find('limit') is not None}
names=json.loads((root.parent/'contact_control_calibration_20261021/wrap04/audit.json').read_text())['joint_names'];scale=np.array([limits[n] for n in names if n in limits]);assert len(scale)==21
s=torch.load(run/'ppo_final.pth',map_location='cpu',weights_only=False)['model']
w=torch.load(run/'critic_ready.pth',map_location='cpu',weights_only=False)['model']
i=torch.load(run/'bc_initial.pth',map_location='cpu',weights_only=False)['model']
def mlp(ob,state,which):
 x=ob
 for j in (0,2,4,6):x=F.elu(F.linear(x,state[f'a2c_network.{which}_mlp.{j}.weight'],state[f'a2c_network.{which}_mlp.{j}.bias']))
 key='mu' if which=='actor' else 'value'
 return F.linear(x,state[f'a2c_network.{key}.weight'],state[f'a2c_network.{key}.bias'])
def evaluate(array,state,kind):
 rows=[]
 with torch.no_grad():
  for chunk in np.array_split(array.reshape(-1,67),max(1,len(array.reshape(-1,67))//1024)):
   y=mlp(torch.from_numpy(chunk),state,kind)
   if kind=='actor':y=.02*y
   else:y=y.clamp(-5,5)*torch.sqrt(state['value_mean_std.running_var'].float()+1e-5)+state['value_mean_std.running_mean'].float()
   rows.append(y.numpy())
 return np.concatenate(rows).reshape(*array.shape[:-1],-1)
def stats(x):
 return dict(max_abs=float(np.max(np.abs(x))),rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.quantile(np.abs(x),.95)))
def first_time(x,threshold):
 mask=x>threshold
 return float(np.nonzero(mask)[0][0]/60) if mask.any() else None
residual=evaluate(b['obs'],s,'actor');residual_actual=evaluate(c['obs'],s,'actor')
qdiff=c['q']-b['q'];wdiff=(c['wrist']-b['wrist'])*1000;odiff=(c['object'][...,:3]-b['object'][...,:3])*1000
report=dict(same_state_learned_correction=dict(joint_rad=stats(residual[...,:21]*scale),wrist_mm=stats(residual[...,21:]*250)),on_updated_rollout_correction=dict(joint_rad=stats(residual_actual[...,:21]*scale),wrist_mm=stats(residual_actual[...,21:]*250)),paired_trajectory_difference=dict(joint_rad=stats(qdiff),wrist_target_mm=stats(wdiff),object_position_mm=stats(odiff)),first_threshold_crossing_s=dict(joint_0_001rad=first_time(np.max(abs(qdiff),axis=(1,2)),.001),wrist_0_05mm=first_time(np.max(abs(wdiff),axis=(1,2)),.05),object_0_1mm=first_time(np.max(abs(odiff),axis=(1,2)),.1)))
returns=np.zeros_like(b['reward']);future=np.zeros(returns.shape[1]);gamma=cfg['gamma'];reward_scale=cfg['reward_shaper']['scale_value']
for t in range(len(returns)-1,-1,-1):
 future=b['reward'][t]*reward_scale+gamma*future*(~b['done'][t]);returns[t]=future
v0=evaluate(b['obs'],i,'critic')[...,0];vw=evaluate(b['obs'],w,'critic')[...,0]
assert np.max(abs(v0-b['value'][...,0]))<1e-4,'Offline value reconstruction differs from actual model'
report['critic_on_reference_mean_policy']=dict(scope='Calibration against a deterministic BC rollout; not an unbiased estimate of the stochastic-policy value or action advantage.',before_rmse=float(np.sqrt(np.mean((v0-returns)**2))),after_warmup_rmse=float(np.sqrt(np.mean((vw-returns)**2))),samples=[dict(t_s=t/60,predicted_mean_before=float(v0[t].mean()),predicted_mean_warm=float(vw[t].mean()),realized_discounted_return_mean=float(returns[t].mean())) for t in (0,300,600,900,1200)],gamma=gamma,reward_scale=reward_scale)
bp=json.loads((run/'before_ppo.json').read_text())['passed'];cp=json.loads((run/'after_ppo.json').read_text())['passed']
report['lost_success_envs']=[j for j,(x,y) in enumerate(zip(bp,cp)) if x and not y];report['gained_success_envs']=[j for j,(x,y) in enumerate(zip(bp,cp)) if y and not x]
report['time_slices']=[dict(t_s=t/60,joint_delta_rad=float(np.max(abs(qdiff[t]))),wrist_delta_mm=float(np.max(abs(wdiff[t]))),object_delta_mm=float(np.max(abs(odiff[t]))),before_holding=int((b['hold'][t]>=3).sum()),after_holding=int((c['hold'][t]>=3).sum())) for t in (0,45,300,600,900,1200,1499)]
report['limits']='One update does not establish long-run PPO stability. Paired rollouts differ through physical feedback; same-state corrections isolate the direct policy change. Thresholds are diagnostic markers, not task termination rules.'
(run/'single_update_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
