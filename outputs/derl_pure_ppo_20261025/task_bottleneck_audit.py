"""First-episode, pre-reset diagnostic stages and discounted-reward audit."""
import json,os,time
from pathlib import Path
import torch
from native_goal_ppo import Policy,Moments,save_json

def run(env,args,out):
 e=env.raw;n=e.num_envs;device=e.device;c=e.graph_control
 kind=os.environ['CONTACT_DIAGNOSTIC'];source=Path(os.environ.get('AUDIT_SOURCE_DIR',out))
 if kind=='forage_ppo':
  from native_goal_ppo import run as train
  args.episode_steps=e.max_episode_length;args.smoke=True;args.steps=32768;args.source=e.hand.cfg.spawn.asset_path
  train(env,args,out)
  save_json(out/'completed.json',dict(status='completed',diagnostic=True,ppo_steps=32768))
  return
 if kind=='strike_collision':trials=[('neutral',None),('final_fixed_wrist','final.pth'),('final_sample','final.pth')]
 elif kind=='forage_surface':trials=[('neutral',None),('random',None)]
 else:
  trials=[('initial_sample','random_initial.pth'),('final_sample','final.pth'),('final_mean','final.pth')]
  if env.task=='strike':trials.append(('final_fixed_wrist','final.pth'))
 tactile=None;tool_view=None
 if kind=='strike_collision':
  from whole_hand_tactile import WholeHandTactile
  tactile=WholeHandTactile(e,'strike',out)
  tool_view=tactile.sim.create_rigid_contact_view(e.cone.cfg.prim_path.replace('.*','*'),filter_patterns=[e.tool_support.cfg.prim_path.replace('.*','*'),e.strike_object.cfg.prim_path.replace('.*','*')])
 rows=[];active={};original=e._get_rewards
 def capture():
  rew=original()
  if not active:return rew
  a=active;live=a['alive'];step=a['step'];metrics={}
  if env.task=='branch':
   t=e.whole_hand_tactile;f=t.digit_forces();threshold=e.cfg.branch_contact_force_threshold
   contact=f>=threshold;op=t.opposition_cosine()
   metrics.update(thumb=contact[:,0],two_long=contact[:,1:].sum(-1)>=2)
   metrics['simultaneous']=metrics['thumb']&metrics['two_long'];metrics['opposed']=metrics['simultaneous']&(op<0)
   metrics['qualified_hold']=e.tactile_streak>=e.cfg.branch_success_hold_steps
   streak=e.tactile_streak;speed=torch.zeros(n,device=device)
   # Count distal vs proximal contact without depending on a fixed joint count.
   distal=[t.names.index(name) for name in e.cartesian_ik.tip_body_names] if hasattr(e.cartesian_ik,'tip_body_names') else [e.hand.body_names.index(name) for name in e.cfg.fingertip_body_names]
   a['distal_peak']=max(a['distal_peak'],float(t.force[live][:,distal,0].norm(dim=-1).max()) if live.any() else 0.)
   bodyforce=t.force[:,:,0].norm(dim=-1)
   a['body_peak']=torch.maximum(a['body_peak'],torch.where(live[:,None],bodyforce,0).amax(0))
   a['contact_only_hold']=torch.maximum(a['contact_only_hold'],torch.where(live,e.tactile_contact_only_streak,0))
  elif env.task=='strike':
   f=e.tool_finger_contact_forces;contact=f>=e.cfg.tool_finger_contact_force_threshold
   metrics.update(thumb=contact[:,0],two_long=contact[:,1:].sum(-1)>=2)
   metrics['simultaneous']=metrics['thumb']&metrics['two_long'];metrics['native_grasp']=e.tool_grasp_streak>=e.cfg.tool_grasp_hold_steps
   from isaaclab.utils.math import matrix_from_quat
   half=torch.tensor(e.cone.cfg.spawn.size,device=device)/2
   extent=(matrix_from_quat(e.cone.data.root_quat_w)[:,2,:].abs()*half).sum(-1)
   lift=e.cone.data.root_pos_w[:,2]-extent-(env.initial_tool_z-half[2])
   speed=e.cone.data.root_lin_vel_w.norm(dim=-1)
   if tactile is not None:
    tactile.update();body=tactile.force[:,:,0].norm(dim=-1)
    targets=tool_view.get_contact_force_matrix(dt=e.cfg.sim.dt).reshape(n,2,3)
    a['body_peak']=torch.maximum(a['body_peak'],torch.where(live[:,None],body,0).amax(0))
    for i in torch.where(live&(speed>10)&(speed>a['speed']))[0].tolist():
     a['collision_events'].append(dict(env=i,step=step,speed=float(speed[i]),previous_tool=a['previous_tool'][i].tolist(),tool=e.cone.data.root_state_w[i].tolist(),support=e.tool_support.data.root_state_w[i].tolist(),target=e.strike_object.data.root_state_w[i].tolist(),wrist=e.graph_wrist_offset[i].tolist(),body_force_N=dict(zip(tactile.names,body[i].tolist())),support_force=targets[i,0].tolist(),target_force=targets[i,1].tolist()))
    a['previous_tool']=e.cone.data.root_state_w.clone()
   metrics.update(lift10mm=lift>.01,held_lift=metrics['native_grasp']&(lift>.01),slow_held_lift=metrics['native_grasp']&(lift>.01)&(speed<.15))
   streak=env.stage_hold
   a['native_holdmax']=torch.maximum(a['native_holdmax'],torch.where(live,e.tool_grasp_streak,0))
   a['liftmax']=torch.maximum(a['liftmax'],torch.where(live,lift,0))
   lost=live&a['ever'].get('native_grasp',torch.zeros_like(live))&~metrics['simultaneous'];a['lost']|=lost
  else:
   t=e.whole_hand_tactile;t.update();f=t.force[:,:,:2].norm(dim=-1).sum(1)
   speed=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in (e.leaf_one,e.leaf_two)],-1).amax(-1)
   metrics.update(contact_both=e.leaf_contact_seen.all(-1),exposed=env.exposure>=.98,settled=speed<.15)
   streak=e.clear_streak
   z=torch.stack([x.data.root_pos_w[:,2] for x in (e.leaf_one,e.leaf_two)],-1)
   a['z_min']=min(a['z_min'],float(z[live].min()) if live.any() else a['z_min']);a['z_max']=max(a['z_max'],float(z[live].max()) if live.any() else a['z_max'])
   a['exposuremax']=torch.maximum(a['exposuremax'],torch.where(live,env.exposure,0))
  metrics['success']=env.event
  for key,value in metrics.items():
   a['ever'].setdefault(key,torch.zeros_like(live));a['ever'][key]|=live&value
   a['frames'][key]=a['frames'].get(key,0)+int((live&value).sum())
  a['holdmax']=torch.maximum(a['holdmax'],torch.where(live,streak,0))
  a['force']=torch.maximum(a['force'],torch.where(live,f.amax(-1),0));a['speed']=torch.maximum(a['speed'],torch.where(live,speed,0))
  a['discounted']+=live*(.99**step)*rew
  if live.any():a['minimum_reward']=min(a['minimum_reward'],float(rew[live].min()))
  if step%120==0:
   a['trace'].append(dict(step=step,alive=int(live.sum()),mean_phi=float(env.prev_potential[live].mean()) if live.any() else 0.,counts={k:int((v&live).sum()) for k,v in metrics.items()}))
   save_json(out/'audit_progress.json',dict(trial=a['mode'],step=step,updated=time.time(),counts={k:int(v.sum()) for k,v in a['ever'].items()}))
  terminal=live&(e.reset_terminated|e.reset_time_outs)
  a['timeouts']+=int((terminal&e.reset_time_outs).sum());a['early']+=int((terminal&~e.reset_time_outs).sum())
  return rew
 e._get_rewards=capture
 for mode,checkpoint in trials:
  active.clear();env.reset_seed=args.eval_seed;obs=env.reset();torch.manual_seed(args.eval_seed+1)
  _,initial_phi=env.features();zero=torch.zeros(n,device=device,dtype=torch.bool)
  active.update(mode=mode,alive=~zero,step=0,ever={},frames={},holdmax=zero.long(),native_holdmax=zero.long(),contact_only_hold=zero.long(),lost=zero.clone(),length=zero.long(),force=zero.float(),speed=zero.float(),liftmax=zero.float(),exposuremax=zero.float(),discounted=zero.float(),body_peak=torch.zeros(len(e.hand.body_names),device=device),distal_peak=0.,z_min=10.,z_max=-10.,minimum_reward=0.,timeouts=0,early=0,trace=[])
  if tactile is not None:
   tactile.reset();active['previous_tool']=e.cone.data.root_state_w.clone();active['collision_events']=[]
  q=e.hand.data.joint_pos.index_select(-1,c.indices);neutral=torch.zeros_like(e.graph_actions);neutral[:,:len(c.indices)]=2*(q-c.lower)/(c.upper-c.lower)-1
  if checkpoint:
   ck=torch.load(source/checkpoint,map_location=device,weights_only=False)
   net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(device);net.load_state_dict(ck['policy']);net.eval();rms=Moments((env.observation_space.shape[0],),device)
   for k,v in ck['observation_moments'].items():setattr(rms,k,v)
  cap=240 if mode=='neutral' else e.max_episode_length
  for step in range(cap):
   active['step']=step
   with torch.no_grad():
    if mode=='neutral':action=neutral
    elif mode=='random':action=torch.randn_like(neutral)
    else:
     dist,_=net(rms.normalize(obs));action=dist.mean if mode=='final_mean' else dist.sample()
     if mode=='final_fixed_wrist':action[:,-3:]=0
   obs,reward,done,info=env.step(action)
   if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():raise RuntimeError('Nonfinite diagnostic state')
   active['length']+=active['alive'].long();active['alive'] &= ~done
   if not active['alive'].any():break
  a=active;ended=~a['alive'];success=a['ever'].get('success',zero)
  # For failed complete episodes, potential shaping telescopes exactly to -Phi0.
  failure=ended&~success
  row=dict(mode=mode,num_envs=n,counts={k:int(v.sum()) for k,v in a['ever'].items()},frame_fractions={k:v/max(1,int(a['length'].sum())) for k,v in a['frames'].items()},max_hold_steps=int(a['holdmax'].max()),max_native_grasp_steps=int(a['native_holdmax'].max()),max_contact_only_hold=int(a['contact_only_hold'].max()),lost_after_grasp=int(a['lost'].sum()),peak_force_N=float(a['force'].max()),peak_speed_m_s=float(a['speed'].max()),max_bottom_lift_m=float(a['liftmax'].max()),mean_episode_steps=float(a['length'].float().mean()),timeouts=a['timeouts'],early=a['early'],unfinished=int(a['alive'].sum()),failure_discounted_residual_max=float((a['discounted'][failure]+initial_phi[failure]).abs().max()) if failure.any() else None,trace=a['trace'])
  if tactile is not None:row.update(collision_events=a['collision_events'],body_peak_N=dict(zip(tactile.names,a['body_peak'].tolist())))
  if env.task=='branch':row.update(body_peak_N=dict(zip(e.hand.body_names,a['body_peak'].tolist())),distal_peak_N=a['distal_peak'])
  if env.task=='forage':row.update(z_min=a['z_min'],z_max=a['z_max'],mean_max_exposure=float(a['exposuremax'].mean()))
  rows.append(row);save_json(out/'bottleneck_audit.json',dict(task=env.task,trials=rows,source=str(source),same_checkpoint_inputs=True,first_episode_only=True))
  print('AUDIT',env.task,mode,row,flush=True)
 if env.task=='forage':
  neutral=rows[0]
  assert neutral['counts'].get('success',0)==0,'Spontaneous Forage success'
  assert .09<neutral['z_min'] and neutral['z_max']<.12,'Unsupported or unstable neutral leaf reset'
 save_json(out/'completed.json',dict(status='completed',diagnostic=True,num_envs=n))
