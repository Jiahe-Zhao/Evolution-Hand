"""Recovery windows for RL. Keep success definitions and hardware limits unchanged."""
import math

def streak_update(previous,condition):
 import torch
 return torch.where(condition,previous+1,torch.zeros_like(previous))

def install(cls,cfg,task,out):
 import torch,json
 dt=cfg.sim.dt*cfg.decimation
 workspace_steps=max(1,math.ceil(.10/dt));speed_steps=max(1,math.ceil(.15/dt))
 changes={'workspace_exit_grace_s':.10,'forage_speed_grace_s':.15,'physical_limits_unchanged':True,'success_thresholds_unchanged':True,'not_deployed_to_paused_training':True}
 if task=='branch':
  changes['episode_length_s']={'before':cfg.episode_length_s,'after':20.};cfg.episode_length_s=20.
 original_reset=cls._reset_idx;original_dones=cls._get_dones
 def ensure(e):
  if not hasattr(e,'rl_outside_streak'):
   e.rl_outside_streak=torch.zeros(e.num_envs,device=e.device,dtype=torch.long);e.rl_leaf_speed_streak=torch.zeros_like(e.rl_outside_streak)
 def reset(e,ids):
  original_reset(e,ids);ensure(e);ids=slice(None) if ids is None else ids;e.rl_outside_streak[ids]=0;e.rl_leaf_speed_streak[ids]=0
 cls._reset_idx=reset
 if task=='forage':
  original_rewards=cls._get_rewards
  def rewards(e):
   ensure(e)
   # Historical implementation latches a one-frame speed excursion forever.
   # Clear that latch before computing this step; invalid frames still reset success hold.
   e.invalid_leaf_motion.zero_();result=original_rewards(e)
   distances=e._leaf_distances();speeds=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in (e.leaf_one,e.leaf_two)],dim=-1)
   finite=torch.isfinite(distances).all(-1)&torch.isfinite(speeds).all(-1)
   far=(distances>e.cfg.maximum_valid_leaf_distance).any(-1)
   e.rl_leaf_speed_streak=streak_update(e.rl_leaf_speed_streak,(speeds>e.cfg.maximum_valid_leaf_speed).any(-1))
   e.invalid_leaf_motion[:]=~finite|far|(e.rl_leaf_speed_streak>=speed_steps)
   return result
  cls._get_rewards=rewards
 def dones(e):
  ensure(e);terminated,truncated=original_dones(e)
  if task=='branch':return terminated,truncated
  if task=='forage':
   outside=e.food_pos[:,:2].norm(dim=-1)>.22;success=e.clear_streak>=e.cfg.leaf_success_hold_steps
   immediate=e.invalid_leaf_motion|~torch.isfinite(e.food_pos).all(-1)
  elif task=='strike':
   outside=(e.cone_pos[:,:2]-e.strike_target_pos[:,:2]).norm(dim=-1)>=e.cfg.workspace_xy_radius
   outside|=(e.cone_pos[:,2]<=e.cfg.workspace_min_height)|(e.cone_pos[:,2]>=e.cfg.workspace_max_height)
   success=e.reset_goal_buf.bool();immediate=~torch.isfinite(e.cone.data.root_state_w).all(-1)
  else:
   outside=(e.grasp_object_pos-e.in_hand_pos).norm(dim=-1)>=e.cfg.fall_dist
   success=torch.zeros_like(outside);immediate=~torch.isfinite(e.grasp_object.data.root_state_w).all(-1)
  immediate|=~torch.isfinite(e.hand.data.joint_pos).all(-1)
  e.rl_outside_streak=streak_update(e.rl_outside_streak,outside)
  return success|immediate|(e.rl_outside_streak>=workspace_steps),truncated
 cls._get_dones=dones
 (out/'rl_tolerance_contract.json').write_text(json.dumps(changes,indent=2))
