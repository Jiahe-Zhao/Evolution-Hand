"""Validate reconstructed configs and contact semantics before Shadow integration."""
from pathlib import Path
import json
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import torch
from strike_shared_scene import load_contract,rigid_object_from_contract,native_hit,shadow_digit_groups
b=Path(__file__).resolve().parent;out=b/'shadow_xyz_20261029';c=load_contract(b/'strike_50m_pair_20261029/ours/native_physics_config.json');checks={}
for name in ['Cone_cfg','strike_object_cfg']:
 config=rigid_object_from_contract(c[name],'/World/'+name).to_dict()
 for field in ['size','rigid_props','collision_props','mass_props','physics_material']:
  original=c[name]['spawn'].get(field);new=config['spawn'].get(field)
  if isinstance(original,dict):original={k:v for k,v in original.items() if k!='func'}
  if isinstance(new,dict):new={k:v for k,v in new.items() if k!='func'}
  if isinstance(original,list):new=list(new)
  checks[name+'.'+field]=original==new
# Target impacted by hand alone must not count; XY threshold is strictly '<'.
held=torch.tensor([True,True,False,True]);now=torch.tensor([True,False,True,True]);tip=torch.zeros(4,3);target=tip.clone();force=torch.tensor([[0.,0.,0.],[0.,0.,10.],[0.,0.,10.],[0.,0.,10.]])
checks['current_grasp_and_filtered_impact']=native_hit(held,now,tip,target,force,c).tolist()==[False,False,False,True]
names=json.loads((out/'wrist_test.json').read_text())['body_names'];groups=shadow_digit_groups(names);checks['five_digit_contact_groups']=len(groups)==5 and len(set(i for g in groups for i in g))==sum(map(len,groups))
(out/'shared_scene_checks.json').write_text(json.dumps(dict(passed=all(checks.values()),checks=checks,digit_groups=[[names[i] for i in g] for g in groups],scope='configuration and contact-predicate checks only; no loaded-scene or rollout claim'),indent=2))
app.close()
