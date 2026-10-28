"""Bounded episode progress rewards; final task predicates remain unchanged.

History maxima are observed by the policy, reset per episode, and never paid
twice. This intentionally changes the task objective: useful failed attempts
can earn partial credit. It is not potential-based policy-invariant shaping.
"""
import json
from types import MethodType
import torch
from contact_curriculum import CurriculumAdapter


def progress_gain(current,best):
 current=current.clamp(0,1)
 return (current-best).clamp_min(0),torch.maximum(best,current)


def clearance_progress(relative_xy,leaf_half_extent,food_half_extent):
 """Progress toward leaving either edge, including before any target exposure."""
 return (relative_xy.abs()/(leaf_half_extent+food_half_extent).clamp_min(1e-5)).amax(-1).clamp(0,1)


class ProgressAdapter(CurriculumAdapter):
 def __init__(self,env,task,out):
  if task not in ('branch','forage'):raise ValueError('Progress reward supports Branch/Forage only')
  self.best_progress=None;self.current_progress=None
  if task=='branch':
   self.progress_names=['thumb_approach','two_long_approach','thumb_contact','long_contact','simultaneous_contact','opposed_contact','opposed_hold']
   self.progress_weights=[.5,.5,.5,.5,1.,3.,4.]
  else:
   self.progress_names=['leaf_approach','touch_leaf1','touch_leaf2','clearance_leaf1','clearance_leaf2','exposure','settled_hold']
   self.progress_weights=[1.,.5,.5,1.,1.,4.,2.]
  super().__init__(env,task,out)
  r=self.raw;reset=r._reset_idx
  def reset_hook(raw,ids):
   reset(ids);idx=slice(None) if ids is None else ids
   self.best_progress[idx]=self.measure_progress()[idx]
  r._reset_idx=MethodType(reset_hook,r)
  (out/'progress_reward_contract.json').write_text(json.dumps(self.contract(),indent=2))

 def measure_progress(self):
  r=self.raw;tips=r.cartesian_ik.fingertip_positions_world()
  if self.task=='branch':
   from isaaclab.utils.math import quat_apply_inverse
   relative=quat_apply_inverse(r.branch.data.root_quat_w[:,None,:].expand(-1,5,-1),tips-r.branch.data.root_pos_w[:,None,:])
   radial=relative[:,:,:2].norm(dim=-1);axial=relative[:,:,2].abs()
   distance=(torch.relu(radial-r.cfg.branch_cfg.spawn.radius).square()+torch.relu(axial-r.cfg.branch_cfg.spawn.height/2).square()).sqrt()
   near=torch.exp(-distance/.04)
   force=(r.whole_hand_tactile.digit_forces()/r.cfg.branch_contact_force_threshold).clamp(0,1)
   simultaneous=torch.minimum(force[:,0],force[:,1:].topk(2,dim=-1).values[:,-1])
   opposed=simultaneous*(-r.whole_hand_tactile.opposition_cosine()).clamp(0,1)
   return torch.stack((near[:,0],near[:,1:].topk(2,dim=-1).values.mean(-1),force[:,0],force[:,1:].topk(2,dim=-1).values.mean(-1),simultaneous,opposed,(r.tactile_streak/r.cfg.branch_success_hold_steps).clamp(0,1)),-1)
  from isaaclab.utils.math import matrix_from_quat
  food_half=torch.tensor(r.cfg.food_cfg.spawn.size[:2],device=r.device)/2
  leaves=(r.leaf_one,r.leaf_two);positions=torch.stack([a.data.root_pos_w for a in leaves],1)
  reach=torch.exp(-torch.cdist(tips,positions).amin(1)/.08).mean(-1)
  seen=r.leaf_contact_seen.float();clear=[]
  for i,a in enumerate(leaves):
   extent=(matrix_from_quat(a.data.root_quat_w)[:,:2,:].abs()*torch.tensor(a.cfg.spawn.size,device=r.device)[None,None,:]/2).sum(-1)
   progress=clearance_progress(a.data.root_pos_w[:,:2]-r.food.data.root_pos_w[:,:2],extent,food_half)
   settle=1/(1+(a.data.root_lin_vel_w.norm(dim=-1)/.3).square())
   clear.append(progress*seen[:,i]*settle)
  exposure=self.coverage(r)*seen.all(-1).float()
  return torch.stack((reach,seen[:,0],seen[:,1],*clear,exposure,(r.clear_streak/30).clamp(0,1)),-1)

 def features(self):
  extra,phi=super().features();current=self.measure_progress()
  if self.best_progress is None:self.best_progress=current.clone()
  self.current_progress=current
  return torch.cat((extra,current,self.best_progress),-1),phi

 def task_reward(self,potential,before,terminal,sparse):
  gain,self.best_progress=progress_gain(self.current_progress,self.best_progress)
  weights=torch.tensor(self.progress_weights,device=self.device)
  partial=(gain*weights).sum(-1)
  force=self.raw.whole_hand_tactile.force.norm(dim=-1).flatten(1).amax(-1)
  force_cost=((force-40).clamp_min(0)/40).square().clamp_max(5)
  speed_cost=torch.zeros_like(force_cost)
  if self.task=='forage':
   speed=torch.stack([a.data.root_lin_vel_w.norm(dim=-1) for a in (self.raw.leaf_one,self.raw.leaf_two)],-1).amax(-1)
   speed_cost=torch.relu(speed-1.).square().clamp_max(5)
  cost=self.raw.step_dt*(.01+.05*force_cost+.05*speed_cost)
  reward=partial+10*self.event.float()-cost
  self.reward_terms=dict(partial=float(partial.mean()),success_bonus=float(10*self.event.float().mean()),soft_cost=float(cost.mean()),progress_mean=dict(zip(self.progress_names,self.current_progress.mean(0).tolist())),episode_best_mean=dict(zip(self.progress_names,self.best_progress.mean(0).tolist())))
  return reward

 def contract(self):
  c=super().contract();c.update(reward_version='bounded_progress_v1',shaping='positive improvement of per-component episode maximum; no terminal cancellation',progress_names=self.progress_names,progress_weights=self.progress_weights,partial_reward_upper_bound=sum(self.progress_weights),success_bonus=10.,history_observed=True,success_predicate_unchanged=True,control_unchanged=True,soft_costs='per-second 0.01 time + 0.05 capped overforce cost above 40 N + Forage 0.05 capped overspeed cost above 1 m/s; no new episode termination')
  return c
