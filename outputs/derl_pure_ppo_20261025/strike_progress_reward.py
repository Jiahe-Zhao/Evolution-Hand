"""Full native Strike task with bounded progress, not grasp-stage completion."""
import json,torch
from types import MethodType
from native_goal_adapter import GoalAdapter
from contact_progress_reward import progress_gain

class StrikeProgressAdapter(GoalAdapter):
 def __init__(self,env,task,out):
  self.best_progress=None;self.current_progress=None
  self.progress_names=['reach','thumb_contact','two_long_contact','held','lift','target_alignment','held_target_approach']
  self.progress_weights=[1.,.5,.5,2.,2.,1.,3.]
  self.initial_bottom=None
  super().__init__(env,task,out)
  r=self.raw;reset=r._reset_idx
  def reset_hook(raw,ids):
   reset(ids);idx=slice(None) if ids is None else ids
   self.initial_bottom[idx]=self.bottom()[idx]
   self.best_progress[idx]=self.measure()[idx]
  r._reset_idx=MethodType(reset_hook,r)
  (out/'strike_progress_contract.json').write_text(json.dumps(self.contract(),indent=2))
 def bottom(self):
  from isaaclab.utils.math import matrix_from_quat
  r=self.raw;half=torch.tensor(r.cone.cfg.spawn.size,device=r.device)/2
  return r.cone.data.root_pos_w[:,2]-(matrix_from_quat(r.cone.data.root_quat_w)[:,2,:].abs()*half).sum(-1)
 def measure(self):
  r=self.raw
  if self.initial_bottom is None:self.initial_bottom=self.bottom().clone()
  f=(r.tool_finger_contact_forces/r.cfg.tool_finger_contact_force_threshold).clamp(0,1)
  near=torch.exp(-(r.cartesian_ik.fingertip_positions_world()-r.cone.data.root_pos_w[:,None]).norm(dim=-1)/.08).mean(-1)
  held=(r.tool_grasp_streak/r.cfg.tool_grasp_hold_steps).clamp(0,1)
  current=(f[:,0]>=1)&((f[:,1:]>=1).sum(-1)>=r.cfg.min_long_finger_contacts)
  lift=((self.bottom()-self.initial_bottom)/.01).clamp(0,1)*current.float()
  error=r.strike_target_pos-r.cone_tip_pos
  align=torch.exp(-error[:,:2].norm(dim=-1)/.05)*current.float()
  approach=torch.exp(-error.norm(dim=-1)/.05)*current.float()
  return torch.stack((near,f[:,0],f[:,1:].topk(2,dim=-1).values[:,-1],held,lift,align,approach),-1)
 def features(self):
  extra,phi=super().features();current=self.measure()
  if self.best_progress is None:self.best_progress=current.clone()
  self.current_progress=current
  return torch.cat((extra,current,self.best_progress),-1),phi
 def task_reward(self,potential,before,terminal,sparse):
  gain,self.best_progress=progress_gain(self.current_progress,self.best_progress)
  partial=(gain*torch.tensor(self.progress_weights,device=self.device)).sum(-1)
  force=self.raw.whole_hand_tactile.force.norm(dim=-1).flatten(1).amax(-1)
  cost=self.raw.step_dt*(.01+.05*((force-40).clamp_min(0)/40).square().clamp_max(5))
  self.reward_terms=dict(partial=float(partial.mean()),success_bonus=float((10*self.event.float()).mean()),soft_cost=float(cost.mean()),progress_mean=dict(zip(self.progress_names,self.current_progress.mean(0).tolist())))
  return partial+10*self.event.float()-cost
 def contract(self):
  c=super().contract();c.update(reward_version='strike_bounded_progress_v1',progress_names=self.progress_names,progress_weights=self.progress_weights,partial_reward_upper_bound=sum(self.progress_weights),success_bonus=10.,history_observed=True,success='Native currently-held tool impact only; grasp/lift never terminate successfully',shaping='Positive gain in observed episode maxima; no terminal cancellation',bc=False)
  return c
