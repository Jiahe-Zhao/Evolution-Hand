"""Goal observations and potential shaping around native full-task success events.

No trajectory, teacher, BC override or hand geometry change. All snapshots are
captured inside _get_rewards, before DirectRLEnv's automatic reset.
"""
import json
from types import MethodType
import gymnasium as gym
import torch

class GoalAdapter:
 def __init__(self,env,task,out):
  self.env=env;self.raw=r=env.unwrapped;self.task=task;self.num_envs=r.num_envs;self.device=r.device
  self.goals=self;self.evaluating=False
  self.action_space=gym.spaces.Box(-1,1,(int(r.cfg.action_space),))
  base=r._get_observations()['policy'];extra,_=self.features()
  self.observation_space=gym.spaces.Box(-float('inf'),float('inf'),(base.shape[-1]+extra.shape[-1],))
  self.reward_terms={};self.peak_lift=torch.zeros(self.num_envs,device=self.device)
  self.prev_potential=None;self.event=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  self.held=self.event.clone();self.max_force=0.;self.max_speed=0.;self.max_limit=0.;self.episodes=0
  original=r._get_rewards
  def reward_hook(raw):
   sparse=original();extra,potential=self.features()
   # Current tasks emit >=1000 exclusively for their final native success.
   self.event=(sparse>=999).clone()
   before=potential if self.prev_potential is None else self.prev_potential
   raw.reset_terminated |= self.event
   raw.reset_buf |= self.event
   terminal=(raw.reset_terminated|raw.reset_time_outs).bool()
   next_phi=torch.where(terminal,torch.zeros_like(potential),potential)
   shaping=.99*next_phi-before
   self.prev_potential=potential.detach().clone()
   reward=shaping+10*self.event.float()
   self.reward_terms={'potential':float(potential.mean()),'shaping':float(shaping.mean()),'native_reward':float(sparse.mean())}
   self.last_native=sparse.clone()
   control=raw.graph_control
   q=raw.hand.data.joint_pos.index_select(-1,control.indices)
   self.max_limit=max(self.max_limit,float(torch.maximum(control.lower-q,q-control.upper).clamp_min(0).max()))
   if self.task=='strike':
    force=raw.tool_finger_contact_forces
    assets=(raw.cone,)
   else:
    sensor=raw.branch_contact_sensor if self.task=='branch' else raw.leaf_contact
    force=torch.linalg.vector_norm(sensor.data.force_matrix_w,dim=-1).sum(1)
    assets=(raw.branch,) if self.task=='branch' else (raw.leaf_one,raw.leaf_two)
   self.max_force=max(self.max_force,float(force.max()))
   self.max_speed=max(self.max_speed,max(float(a.data.root_lin_vel_w.norm(dim=-1).max()) for a in assets))
   if not torch.isfinite(extra).all() or not torch.isfinite(reward).all():raise RuntimeError('Nonfinite native goal state/reward')
   return reward
  r._get_rewards=MethodType(reward_hook,r)
  (out/'goal_contract.json').write_text(json.dumps(self.contract(),indent=2))
 def contract(self):
  return dict(task=self.task,version=2,goal_source='native scene state and native success thresholds',
   goals={'branch':'branch center, contact opposition and stable hold duration','forage':'both leaf centers outside food clearance radius with valid hand contact','strike':'tool tip at strike target with impact threshold while currently held'}[self.task],
   success='native final event only; grasp milestone is not Strike success',shaping='gamma*Phi(next)-Phi(current); terminal Phi=0',bc=False,
   observation_size=self.observation_space.shape[0],action_size=self.action_space.shape[0],episode_steps=self.raw.max_episode_length)
 def features(self):
  r=self.raw;n=self.num_envs;d=self.device
  def constant(v):return torch.full((n,1),float(v),device=d)
  if self.task=='branch':
   target=r.branch.data.root_pos_w-r.scene.env_origins
   rel=r.branch.data.root_pos_w-r.hand.data.root_pos_w
   matrix=r.branch_contact_sensor.data.force_matrix_w
   forces=torch.linalg.vector_norm(matrix[:,0,:,:],dim=-1)
   contact=(forces>=r.cfg.branch_contact_force_threshold).float()
   hold=(r.branch_success_streak.float()/r.cfg.branch_success_hold_steps).clamp(0,1)
   self.held=(contact[:,0]>0)&(contact[:,1:].sum(-1)>=r.cfg.min_long_finger_contacts)
   tips=r.cartesian_ik.fingertip_positions_world()
   reach=torch.exp(-torch.linalg.vector_norm(tips-r.branch.data.root_pos_w[:,None,:],dim=-1)/.08).mean(-1)
   phi=reach+contact[:,0]+contact[:,1:].mean(-1)+hold
   extra=torch.cat((target/.1,rel/.1,contact,hold[:,None],constant(r.cfg.branch_success_hold_steps*r.step_dt),constant(r.cfg.branch_contact_force_threshold)),1)
  elif self.task=='forage':
   food=r.food.data.root_pos_w-r.scene.env_origins
   relative=torch.stack((r.leaf_one.data.root_pos_w-r.food.data.root_pos_w,r.leaf_two.data.root_pos_w-r.food.data.root_pos_w),1)
   dist=torch.linalg.vector_norm(relative[:,:,:2],dim=-1)
   desired=relative[:,:,:2]/dist[:,:,None].clamp_min(1e-6)*r.cfg.leaf_clear_distance
   error=desired-relative[:,:,:2]
   contacts=r.leaf_contact_seen.float()
   hold=(r.clear_streak.float()/r.cfg.leaf_success_hold_steps).clamp(0,1)
   self.held=contacts.all(-1)
   tips=r.cartesian_ik.fingertip_positions_world()
   leaves=torch.stack((r.leaf_one.data.root_pos_w,r.leaf_two.data.root_pos_w),1)
   reach=torch.exp(-torch.cdist(tips,leaves).amin(1)/.08).mean(-1)
   phi=reach+(dist/r.cfg.leaf_clear_distance).clamp(0,1).mean(-1)+contacts.mean(-1)+hold
   extra=torch.cat((food/.1,desired.flatten(1)/.1,error.flatten(1)/.1,contacts,hold[:,None],constant(r.cfg.leaf_clear_distance),r.invalid_leaf_motion[:,None].float()),1)
  else:
   target=r.strike_target_pos
   error=target-r.cone_tip_pos
   forces=r.tool_finger_contact_forces
   contact=(forces>=r.cfg.tool_finger_contact_force_threshold).float()
   self.held=(contact[:,0]>0)&(contact[:,1:].sum(-1)>=r.cfg.min_long_finger_contacts)
   impact=torch.linalg.vector_norm(r.strike_object_force,dim=-1)
   distance=torch.linalg.vector_norm(error[:,:2],dim=-1)
   tips=r.cartesian_ik.fingertip_positions_world()
   reach=torch.exp(-torch.linalg.vector_norm(tips-r.cone.data.root_pos_w[:,None,:],dim=-1)/.08).mean(-1)
   phi=reach+contact[:,0]+contact[:,1:].mean(-1)+self.held.float()*torch.exp(-torch.linalg.vector_norm(error,dim=-1)/.05)
   extra=torch.cat((target/.1,error/.1,contact,self.held[:,None].float(),(impact/r.cfg.success_force_threshold).clamp(0,10)[:,None],constant(r.cfg.success_distance),constant(r.cfg.success_force_threshold)),1)
  return extra,phi
 def reset(self):
  obs,_=self.env.reset(seed=getattr(self,'reset_seed',None));extra,phi=self.features();self.prev_potential=phi.detach().clone()
  return torch.cat((obs['policy'],extra),1)
 def step(self,action):
  obs,reward,term,trunc,info=self.env.step(action.clamp(-1,1));done=term|trunc
  event=self.event.clone();held=self.held.clone();extra,phi=self.features()
  self.prev_potential[done]=phi[done].detach()
  self.episodes+=int(done.sum())
  return torch.cat((obs['policy'],extra),1),reward,done,{'episode_success':event,'held':held}
 def diagnostics(self):
  return {'episodes':self.episodes,'max_digit_force_N':self.max_force,'max_speed_m_s':self.max_speed,'max_limit_rad':self.max_limit,'physical_extrema_collected':True}
