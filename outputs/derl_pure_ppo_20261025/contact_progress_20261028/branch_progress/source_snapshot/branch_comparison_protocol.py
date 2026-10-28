"""Reset clearance, matched scene randomness and physical comparison protocol."""
import os,json,hashlib
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from native_goal_ppo import save_json

def install(e,out,mode):
 from branch_comparison_geometry import CollisionCache
 e.comparison_plan=json.loads(Path(os.environ['BRANCH_COMPARISON_MANIFEST']).read_text())
 p=e.comparison_plan
 if p['variant']!=e.variant:raise ValueError('Wrong hand manifest')
 e.comparison_cache=CollisionCache(e)
 e.comparison_valid=torch.ones(e.num_envs,device=e.device,dtype=torch.bool)
 e.comparison_reset_counts=np.zeros(e.num_envs,dtype=np.int64);e.comparison_scene_seed=e.cfg.seed
 e.comparison_initial_collisions={}
 e.comparison_reset_stats=dict(episodes=0,resampled_episodes=0,rejected_joint_samples=0)
 e.jitter=0.;e.placement=None # Placement is applied by the checked reset hook below.
 old_reset=e._reset_idx;old_public_reset=e.reset
 def public_reset(raw,*args,seed=None,**kwargs):
  if seed is not None:
   raw.comparison_scene_seed=int(seed);raw.comparison_reset_counts[:]=0
  return old_public_reset(*args,seed=seed,**kwargs)
 e.reset=MethodType(public_reset,e)
 def reset(raw,ids):
  old_reset(ids)
  if ids is None:ids=torch.arange(raw.num_envs,device=raw.device)
  indices=ids.tolist();state=raw.branch.data.root_state_w[ids].clone()
  centers=[]
  for i in indices:
   if mode=='replay':center=np.asarray(p['centers_world'][i]);jitter=np.zeros(3)
   else:
    center=np.asarray(p['center_world'])
    rng=np.random.default_rng(np.random.SeedSequence([raw.comparison_scene_seed,i,int(raw.comparison_reset_counts[i])]))
    jitter=np.asarray(p['anatomical_rotation'])@rng.uniform(-.002,.002,3)
   raw.comparison_reset_counts[i]+=1
   centers.append(center+jitter)
  state[:,:3]=raw.scene.env_origins[ids]+torch.tensor(np.asarray(centers),device=raw.device,dtype=state.dtype)
  state[:,3:7]=torch.tensor(p['quat_world'],device=raw.device,dtype=state.dtype);state[:,7:]=0
  for j,i in enumerate(indices):
   hits=raw.comparison_cache.collisions(i,state[j,:3].cpu().numpy(),p['quat_world'])
   retries=0
   # Condition reset proposals on nonpenetration; no rollout action restriction.
   # Proposal seeds match across hands, accepted proposals can differ.
   while hits and mode!='replay' and retries<32:
    retries+=1;one=torch.tensor([i],device=raw.device)
    rng=np.random.default_rng(np.random.SeedSequence([raw.comparison_scene_seed,i,int(raw.comparison_reset_counts[i])-1,retries]))
    center=np.asarray(p['center_world'])+np.asarray(p['anatomical_rotation'])@rng.uniform(-.002,.002,3)
    state[j,:3]=raw.scene.env_origins[i]+torch.tensor(center,device=raw.device,dtype=state.dtype)
    q=raw.hand.data.default_joint_pos[one].clone()
    q[:,raw.indices]+=(torch.rand(1,len(raw.indices),device=raw.device)*2-1)*raw.cfg.reset_dof_pos_noise*(raw.upper[raw.indices]-raw.lower[raw.indices])
    q=q.clamp(raw.lower,raw.upper);raw.hand.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=one)
    raw.prev_targets[one]=q;raw.hand.set_joint_position_target(q,env_ids=one);raw.sim.forward();raw.scene.update(0.)
    hits=raw.comparison_cache.collisions(i,state[j,:3].cpu().numpy(),p['quat_world'])
   raw.comparison_reset_stats['episodes']+=1;raw.comparison_reset_stats['resampled_episodes']+=int(retries>0);raw.comparison_reset_stats['rejected_joint_samples']+=retries
   raw.comparison_valid[i]=not hits
   if hits:
    if mode!='replay':raise RuntimeError(f'Initial branch overlap {i}: {hits}')
    raw.comparison_initial_collisions[str(i)]=hits
    state[j,:3]=raw.scene.env_origins[i]+torch.tensor([0.,0.,.1],device=raw.device)
  raw.branch.write_root_state_to_sim(state,env_ids=ids)
  save_json(out/'reset_sampling.json',raw.comparison_reset_stats)
  if mode=='replay':save_json(out/'initial_clearance.json',dict(valid=raw.comparison_valid.tolist(),collisions=raw.comparison_initial_collisions))
  raw.whole_hand_tactile.reset(ids)
  pos,quat=raw._branch_relative_pose();raw.tactile_prev_pos[ids]=pos[ids];raw.tactile_prev_quat[ids]=quat[ids]
 e._reset_idx=MethodType(reset,e)
 t=e.whole_hand_tactile;t.ready_after_reset=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool);t.raw_force=t.force.clone()
 for name in ('_get_rewards','_get_observations'):
  original=getattr(e,name)
  def guarded(raw,*args,_original=original,**kwargs):
   t.update();raw.branch_contact_sensor.data.force_matrix_w[~t.ready_after_reset]=0
   return _original(*args,**kwargs)
  setattr(e,name,MethodType(guarded,e))
 save_json(out/'placement_contract.json',dict(mode=mode,plan=p,initial_clearance_margin_m=.0005,contact_freshness='require a fresh all-clear frame after every reset',jitter_m=0. if mode=='replay' else .002,scenario_rng='SeedSequence(scene_seed,env_id,episode_counter,attempt); matched proposals, acceptance conditioned on per-hand geometric clearance; not exactly paired accepted states'))

def replay(e,args,out,mode):
 from contact_curriculum import CurriculumAdapter
 env=CurriculumAdapter(e,'branch',out);p=e.comparison_plan;n=e.num_envs;c=e.graph_control
 if p['joint_names']!=e.hand.joint_names:raise ValueError('Joint ordering mismatch')
 goal=torch.tensor(p['joint_target'],device=e.device)
 if goal.ndim==1:goal=goal[None].expand(n,-1)
 if len(goal)!=n:raise ValueError('Candidate batch mismatch')
 active={};original=e._get_rewards
 def reward():
  result=original()
  if active:
   f=e.whole_hand_tactile.digit_forces();a=active
   a['success']|=env.event&e.comparison_valid
   a['hold']=torch.maximum(a['hold'],e.tactile_streak)
   a['force']=torch.maximum(a['force'],f.amax(-1))
   if a['step']<120:a['initial_force']=torch.maximum(a['initial_force'],f.amax(-1));a['initial_success']|=env.event
   if a['step']>=1140:a['release_force']=torch.maximum(a['release_force'],f.amax(-1))
   e.reset_terminated.zero_();e.reset_buf=e.reset_time_outs.clone()
  return result
 e._get_rewards=reward;trials=[]
 for seed in ([1062803] if mode=='replay' else [1062803,1062809]):
  active.clear();env.reset_seed=seed;env.reset();q0=e.hand.data.joint_pos.clone();z=torch.zeros(n,device=e.device,dtype=torch.bool)
  active.update(success=z.clone(),hold=z.long(),force=z.float(),initial_force=z.float(),initial_success=z.clone(),release_force=z.float(),step=0)
  locked=z.clone();hold_target=q0.clone();valid=e.comparison_valid.clone()
  for step in range(e.max_episode_length):
   active['step']=step;target=q0+(goal-q0)*max(0.,min(1.,(step-119)/600))
   new=(e.tactile_streak>=15)&~locked&(step>=120)
   hold_target[new]=target[new];locked|=new;target=torch.where(locked[:,None],hold_target,target)
   if step>=1000:target=q0
   target=torch.where(valid[:,None],target,q0)
   val=target[:,c.indices].clamp(c.lower,c.upper);action=2*(val-c.lower)/(c.upper-c.lower)-1
   try:env.step(action)
   except Exception:
    tensors=dict(q=e.hand.data.joint_pos,dq=e.hand.data.joint_vel,body_pos=e.hand.data.body_pos_w,body_quat=e.hand.data.body_quat_w,branch=e.branch.data.root_state_w,forces=e.whole_hand_tactile.force,action=action)
    save_json(out/'failure_state.json',dict(step=step,nonfinite={k:(~torch.isfinite(v)).nonzero().tolist() for k,v in tensors.items()},tensors={k:v.tolist() for k,v in tensors.items()},active={k:v.tolist() if torch.is_tensor(v) else v for k,v in active.items()}))
    raise
   if step==1190:
    active['release_geometry_collisions']=[e.comparison_cache.collisions(i,e.branch.data.root_pos_w[i].cpu().numpy(),p['quat_world']) for i in range(n)]
   if step%120==0:save_json(out/'audit_progress.json',dict(seed=seed,step=step,successes=int(active['success'].sum()),max_hold=int(active['hold'].max())))
  a=active;sustained=valid&(a['hold']>=180)&~a['initial_success']&(a['initial_force']<1.)
  trial=dict(seed=seed,episodes=n,successes=int(a['success'].sum()),sustained_3s=int(sustained.sum()),valid=valid.tolist(),success=a['success'].tolist(),sustained=sustained.tolist(),hold_steps=a['hold'].tolist(),peak_force_N=a['force'].tolist(),initial_force_N=a['initial_force'].tolist(),initial_success=a['initial_success'].tolist(),release_force_N=a['release_force'].tolist(),released_clear=int((a['release_force']<1).sum()),initial_collisions=e.comparison_initial_collisions,hold_target=hold_target.tolist())
  trial['release_geometry_collisions']=a.get('release_geometry_collisions');trials.append(trial);save_json(out/'script_validation.json',trials)
 save_json(out/'completed.json',dict(status='completed',script_only=True,trials=trials))

def train(e,args,out):
 from contact_curriculum import CurriculumAdapter
 if os.environ.get('CONTACT_PROGRESS_REWARD')=='1':
  from contact_progress_reward import ProgressAdapter as CurriculumAdapter
 from native_goal_ppo import run,Policy
 env=CurriculumAdapter(e,'branch',out)
 original_diagnostics=env.diagnostics
 def diagnostics():
  d=original_diagnostics();d.update(reset_sampling=e.comparison_reset_stats.copy(),fresh_contact_fraction=float(e.whole_hand_tactile.ready_after_reset.float().mean()))
  return d
 env.diagnostics=diagnostics
 args.episode_steps=e.max_episode_length;args.smoke=False;args.source=str(out/'asset_descriptor.json')
 asset=getattr(e.cfg.robot_cfg.spawn,'asset_path',getattr(e.cfg.robot_cfg.spawn,'usd_path',None))
 descriptor=dict(variant=e.variant,asset=asset,robot_cfg=e.cfg.robot_cfg.to_dict(),placement=e.comparison_plan)
 Path(args.source).write_text(json.dumps(descriptor,default=str,indent=2))
 net=Policy(env.observation_space.shape[0],env.action_space.shape[0])
 save_json(out/'comparison_model.json',dict(observation_size=env.observation_space.shape[0],action_size=env.action_space.shape[0],parameter_count=sum(p.numel() for p in net.parameters()),architecture=[64,64],joint_names=e.hand.joint_names,hand_structure_unchanged=True))
 run(env,args,out)
