"""Reset-only branch translation, identical for training and evaluation."""
import hashlib,json,os
from pathlib import Path
from types import MethodType
import torch

def attach(e,out):
 path=Path(os.environ['BRANCH_PLACEMENT_MANIFEST']);plan=json.loads(path.read_text())
 (out/'branch_placement_manifest.json').write_text(json.dumps(plan,indent=2))
 if hashlib.sha256(Path(e.hand.cfg.spawn.asset_path).read_bytes()).hexdigest()!=plan['hand_sha256']:
  raise RuntimeError('Branch placement was not validated for this hand')
 amplitude=float(os.environ.get('BRANCH_POSITION_JITTER_M','0.002'))
 shift=torch.tensor(plan['shift_world_m'],device=e.device)
 old=e._reset_idx
 def reset(raw,ids):
  old(ids);idx=slice(None) if ids is None else ids
  state=raw.branch.data.root_state_w[idx].clone()
  perturb=(torch.rand(len(state),3,device=e.device)*2-1)*amplitude
  state[:,:3]+=shift+perturb
  raw.branch.write_root_state_to_sim(state,ids);raw._compute_intermediate_values()
  verify_initial_clearance(raw,ids)
  p,q=raw._branch_relative_pose();raw.tactile_prev_pos[idx]=p[idx];raw.tactile_prev_quat[idx]=q[idx]
 e._reset_idx=MethodType(reset,e)
 # The reset distribution must be geometrically clear before this freshness
 # guard is allowed. It never treats a missing/old report as a grasp.
 t=e.whole_hand_tactile
 t.ready_after_reset=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
 t.raw_force=t.force.clone()
 for method_name in ('_get_rewards','_get_observations'):
  original=getattr(e,method_name)
  def guarded(raw,*args,_original=original,**kwargs):
   t.update()
   raw.branch_contact_sensor.data.force_matrix_w[~t.ready_after_reset]=0
   return _original(*args,**kwargs)
  setattr(e,method_name,MethodType(guarded,e))
 e.branch_placement_plan=plan
 (out/'branch_placement_contract.json').write_text(json.dumps(dict(manifest=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shift_world_m=plan['shift_world_m'],uniform_position_jitter_each_axis_m=amplitude,hand_geometry_unchanged=True,reset_only=True,identical_train_eval_reset=True,initial_geometry_check="FCL imported collision meshes, 0.5 mm margin",contact_freshness="zero force after reset required before contact credit",sleep_configuration_unchanged=True),indent=2))

def verify_initial_clearance(e,ids):
 import fcl
 from scipy.spatial.transform import Rotation
 from imported_hand_geometry import hand_collision_objects
 indices=range(e.num_envs) if ids is None else ids.tolist()
 for i in indices:
  state=e.branch.data.root_state_w[i].detach().cpu().numpy()
  objects,_=hand_collision_objects(e,i)
  shape=fcl.Cylinder(float(e.cfg.branch_cfg.spawn.radius)+.0005,float(e.cfg.branch_cfg.spawn.height)+.001)
  branch=fcl.CollisionObject(shape,fcl.Transform(Rotation.from_quat(state[[4,5,6,3]]).as_matrix(),state[:3]))
  for name,body in objects:
   result=fcl.CollisionResult();fcl.collide(branch,body,fcl.CollisionRequest(),result)
   if result.is_collision:raise RuntimeError(f'Branch initial geometry collision in env {i}: {name}')


def validate(env,args,out):
 from native_goal_ppo import save_json
 e=env.raw;c=e.graph_control;p=e.branch_placement_plan;n=e.num_envs
 assert list(e.hand.joint_names)==p['joint_names']
 goal=torch.tensor(p['joint_target'],device=e.device)[None].expand(n,-1)
 rows=[];active={};original=e._get_rewards
 def reward():
  r=original()
  if active:
   a=active;f=e.whole_hand_tactile.digit_forces();live=a['alive']
   a['success']|=live&env.event;a['hold']=torch.maximum(a['hold'],torch.where(live,e.tactile_streak,0))
   a['force']=torch.maximum(a['force'],torch.where(live,f.amax(-1),0))
   ready=e.whole_hand_tactile.ready_after_reset
   first=ready&(a['first_fresh']<0);a['first_fresh'][first]=a['step']
   if a['step']>=1140:a['release_force']=torch.maximum(a['release_force'],f.amax(-1))
   if a['step']<120:
    a['backend_initial_force']=torch.maximum(a['backend_initial_force'],e.whole_hand_tactile.raw_force.norm(dim=-1).flatten(1).amax(-1))
    a['initial_raw_force']=torch.maximum(a['initial_raw_force'],f.amax(-1));a['initial_success']|=env.event
    if a['step']>=10:a['initial_force']=torch.maximum(a['initial_force'],f.amax(-1))
   e.reset_terminated.zero_();e.reset_buf=e.reset_time_outs.clone()
  return r
 e._get_rewards=reward
 for seed in [1062803,1062809]:
  active.clear();env.reset_seed=seed;obs=env.reset();q0=e.hand.data.joint_pos.clone();z=torch.zeros(n,device=e.device,dtype=torch.bool)
  save_json(out/f'reset_{seed}.json',dict(joint_pos=q0.tolist(),default_joint_pos=e.hand.data.default_joint_pos.tolist(),prev_targets=e.prev_targets.tolist(),branch_pos=(e.branch.data.root_pos_w-e.scene.env_origins).tolist(),tactile_streak=e.tactile_streak.tolist(),episode_length=e.episode_length_buf.tolist()))
  active.update(backend_initial_force=z.float(),release_force=z.float(),first_fresh=torch.full((n,),-1,device=e.device,dtype=torch.long),alive=~z,success=z.clone(),hold=z.long(),force=z.float(),initial_force=z.float(),initial_raw_force=z.float(),initial_success=z.clone(),step=0)
  locked=z.clone();hold_target=q0.clone()
  for step in range(int(os.environ.get('BRANCH_DIAGNOSTIC_STEPS',e.max_episode_length))):
   active['step']=step;target=q0+(goal-q0)*max(0.,min(1.,(step-119)/p['ramp_steps']))
   new=(e.tactile_streak>=e.cfg.branch_success_hold_steps)&~locked&(step>=120)&(os.environ.get('BRANCH_NO_LOCK')!='1')
   hold_target[new]=target[new];locked|=new
   target=torch.where(locked[:,None],hold_target,target)
   if step>=1000:target=q0
   val=target.index_select(-1,c.indices).clamp(c.lower,c.upper);action=2*(val-c.lower)/(c.upper-c.lower)-1
   obs,rew,done,info=env.step(action);active['alive']&=~done
   if step%120==0:
    save_json(out/'audit_progress.json',dict(seed=seed,step=step))
    with (out/'reset_trace.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,step=step,sim_step=e.sim.current_time_step_index,stamp=e.whole_hand_tactile.stamp,q=e.hand.data.joint_pos[0].tolist(),forces=e.whole_hand_tactile.digit_forces()[0].tolist(),locked=bool(locked[0]),streak=int(e.tactile_streak[0]),native_force=e.branch_contact_sensor.data.force_matrix_w[0].tolist(),physx_q=e.hand.root_physx_view.get_dof_positions()[0].tolist()))+'\n')
   if not active['alive'].any():break
  a=active;valid=(a['hold']>=180)&(a['initial_force']<1.)&~a['initial_success']
  rows.append(dict(seed=seed,episodes=n,successes=int(a['success'].sum()),sustained_3s=int(valid.sum()),initial_collision_envs=int((a['initial_force']>=1).sum()),initial_successes=int(a['initial_success'].sum()),backend_initial_peak_force_N=float(a['backend_initial_force'].max()),first_fresh_contact_frame=a['first_fresh'].tolist(),released_clear=int((a['release_force']<1.).sum()),release_peak_force_N=float(a['release_force'].max()),initial_raw_peak_force_N=float(a['initial_raw_force'].max()),initial_settled_peak_force_N=float(a['initial_force'].max()),max_hold_steps=int(a['hold'].max()),hold_steps=a['hold'].tolist(),peak_force_N=float(a['force'].max())))
  save_json(out/'script_validation.json',rows)
 save_json(out/'completed.json',dict(status='completed',script_only=True,trials=rows))
