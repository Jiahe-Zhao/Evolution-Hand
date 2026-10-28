"""Read-only full rigid-body tactile coverage; no geometry/material changes.

Net body force includes all collisions. Filtered force identifies task objects.
These are resultant-force sensors, not pressure images or contact patch estimates.
"""
import json,re
from types import MethodType
import torch
from isaaclab.utils.math import quat_apply_inverse

class WholeHandTactile:
 def __init__(self,e,task,out):
  import omni.physics.tensors as tensors
  self.e=e;self.task=task;self.names=list(e.hand.body_names);self.n=e.num_envs
  self.sim=tensors.create_simulation_view('torch');self.sim.set_subspace_roots('/')
  objects={'branch':['branch'],'forage':['leaf_one','leaf_two','food'],'strike':['cone'],'grasp':['grasp_object']}[task]
  self.targets=[getattr(e,k) for k in objects]
  self.views=[]
  for name in self.names:
   path=e.hand.cfg.prim_path.replace('.*','*')+'/'+name
   filters=[a.cfg.prim_path.replace('.*','*') for a in self.targets]
   v=self.sim.create_rigid_contact_view(path,filter_patterns=filters)
   if v.sensor_count!=self.n:raise RuntimeError(f'Tactile body count mismatch {name}: {v.sensor_count}')
   self.views.append(v)
  self.groups=[]
  for name in self.names:
   m=re.match(r'link_(\d+)_',name)
   group=getattr(e,'tactile_body_groups',{}).get(name,int(m[1]) if m and int(m[1])>0 else 0)
   self.groups.append(group)
  self.digits=sorted(set(self.groups)-{0});self.stamp=None
  shape=(self.n,len(self.names),len(objects),3)
  self.force=torch.zeros(shape,device=e.device);self.total=torch.zeros(shape[:2]+(3,),device=e.device)
  self.age=torch.zeros(shape[:-1],device=e.device);self.previous=self.force.clone();self.delta=self.force.clone()
  self.manifest=dict(version=1,bodies=self.names,body_digit_groups=self.groups,targets=objects,coverage='all runtime articulation bodies; fixed merged geometry belongs to its surviving rigid body',features='target contact flag, local target force xyz, contact duration, force change xyz; net body force xyz',contact_threshold_N=.05,force_units='N',no_geometry_change=True)
  (out/'whole_hand_tactile_contract.json').write_text(json.dumps(self.manifest,indent=2))
 def reset(self,ids=None):
  ids=slice(None) if ids is None else ids
  if hasattr(self,'ready_after_reset'):self.ready_after_reset[ids]=False
  self.age[ids]=0;self.previous[ids]=0;self.force[ids]=0;self.delta[ids]=0;self.total[ids]=0
  # Tensor contact buffers can still contain the preceding episode until the
  # next physics step. Do not expose those forces in reset observations.
  self.stamp=self.e.sim.current_time_step_index if hasattr(self,'e') else None
 def update(self):
  stamp=self.e.sim.current_time_step_index
  if stamp==self.stamp:return
  self.stamp=stamp
  f=torch.stack([v.get_contact_force_matrix(dt=self.e.cfg.sim.dt).reshape(self.n,len(self.targets),3) for v in self.views],1)
  total=torch.stack([v.get_net_contact_forces(dt=self.e.cfg.sim.dt).reshape(self.n,3) for v in self.views],1)
  if hasattr(self,'ready_after_reset'):
   # Tensor contact reports can retain a pre-teleport contact until motion
   # generates a fresh report. Use only after observing a clear-contact frame.
   self.raw_force=f.clone()
   self.ready_after_reset |= f.norm(dim=-1).flatten(1).amax(-1)<0.05
   f=f*self.ready_after_reset[:,None,None,None]
   total=total*self.ready_after_reset[:,None,None]
  self.delta=f-self.force;self.previous=self.force;self.force=f;self.total=total
  self.age=torch.where(f.norm(dim=-1)>.05,self.age+self.e.step_dt,torch.zeros_like(self.age))
  if not torch.isfinite(f).all():raise RuntimeError('Nonfinite full-hand contact force')
 def digit_forces(self,target=0):
  self.update();f=self.force[:,:,target].norm(dim=-1)
  # Maximum per body avoids granting more force credit merely for more links.
  return torch.stack([f[:,[i for i,g in enumerate(self.groups) if g==d]].amax(-1) for d in self.digits],-1)
 def opposition_cosine(self,target=0):
  self.update()
  vectors=torch.stack([self.force[:,[i for i,g in enumerate(self.groups) if g==d],target].sum(1) for d in self.digits],1)
  thumb=vectors[:,0];other=vectors[:,1:].sum(1)
  return (thumb*other).sum(-1)/(thumb.norm(dim=-1)*other.norm(dim=-1)).clamp_min(1e-8)
 def features(self):
  self.update();q=self.e.hand.data.body_quat_w
  local=quat_apply_inverse(q[:,:,None,:].expand(-1,-1,len(self.targets),-1),self.force)
  change=quat_apply_inverse(q[:,:,None,:].expand(-1,-1,len(self.targets),-1),self.delta)
  total=quat_apply_inverse(q,self.total)
  return torch.cat(((self.force.norm(dim=-1)>.05).float().flatten(1),local.flatten(1)/5,self.age.clamp_max(3).flatten(1)/3,change.flatten(1)/5,total.flatten(1)/5),-1)

def attach(e,task,out,branch_success=False):
 t=WholeHandTactile(e,task,out);e.whole_hand_tactile=t
 reset=e._reset_idx
 def reset_hook(raw,ids):
  reset(ids);t.reset(ids)
 e._reset_idx=MethodType(reset_hook,e)
 if task=='branch' and branch_success:
  (out/'branch_tactile_success_contract.json').write_text(json.dumps(dict(version=2,contact_force_threshold_N=e.cfg.branch_contact_force_threshold,min_long_fingers=e.cfg.min_long_finger_contacts,hold_steps=e.cfg.branch_success_hold_steps,added_requirement='thumb and long-finger force resultants have negative dot product',legacy_fingertip_event_retained=True,contact_only_streak_retained=True),indent=2))
  original=e._get_rewards
  def reward(raw):
   # Retain exact original event for auditing before updating full-hand streak.
   raw.branch_success_streak=raw.tactile_old_streak.clone()
   old=original();raw.tactile_old_event=old.clone()
   raw.tactile_old_streak=raw.branch_success_streak.clone()
   f=t.digit_forces();pos,quat=raw._branch_relative_pose()
   delta=(pos-raw.tactile_prev_pos).norm(dim=-1)
   angle=2*torch.acos((quat*raw.tactile_prev_quat).sum(-1).abs().clamp(max=1))
   stable=(delta<=raw.cfg.branch_relative_position_tolerance)&(angle<=raw.cfg.branch_relative_rotation_tolerance)
   if not raw.cfg.require_pose_stability:stable[:]=True
   qualified=(f[:,0]>=raw.cfg.branch_contact_force_threshold)&((f[:,1:]>=raw.cfg.branch_contact_force_threshold).sum(-1)>=raw.cfg.min_long_finger_contacts)&stable
   raw.tactile_contact_only_streak=torch.where(qualified,raw.tactile_contact_only_streak+1,0)
   # A fixed branch can support same-side pushing forever. A grasp must have
   # opposing thumb/long-finger resultant forces, not merely simultaneous touch.
   raw.tactile_opposition=t.opposition_cosine()
   qualified=qualified&(raw.tactile_opposition<0)
   raw.tactile_streak=torch.where(qualified,raw.tactile_streak+1,0)
   raw.tactile_prev_pos=pos.clone();raw.tactile_prev_quat=quat.clone()
   raw.branch_success_streak=raw.tactile_streak.clone()
   return raw.cfg.success_reward*(raw.tactile_streak==raw.cfg.branch_success_hold_steps).float()
  e.tactile_contact_only_streak=torch.zeros(e.num_envs,dtype=torch.long,device=e.device)
  e.tactile_opposition=torch.zeros(e.num_envs,device=e.device)
  e.tactile_streak=torch.zeros(e.num_envs,dtype=torch.long,device=e.device)
  e.tactile_prev_pos,e.tactile_prev_quat=e._branch_relative_pose()
  # Keep old and new streaks independent when original reward executes.
  e.tactile_old_streak=e.branch_success_streak.clone()
  e._get_rewards=MethodType(reward,e)
  prior_reset=e._reset_idx
  def branch_reset(raw,ids):
   prior_reset(ids);idx=slice(None) if ids is None else ids
   raw.tactile_streak[idx]=0;raw.tactile_old_streak[idx]=0;raw.tactile_contact_only_streak[idx]=0;raw.tactile_opposition[idx]=0
   p,q=raw._branch_relative_pose();raw.tactile_prev_pos[idx]=p[idx];raw.tactile_prev_quat[idx]=q[idx]
  e._reset_idx=MethodType(branch_reset,e)
 return t
