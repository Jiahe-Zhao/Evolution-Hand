"""Unchanged-geometry FK multistart search followed by physical target replay."""
import json,math,time,os,xml.etree.ElementTree as ET
from pathlib import Path
import torch
from native_goal_ppo import save_json

def run(env,args,out):
 import trimesh
 from isaaclab.utils.math import matrix_from_quat
 e=env.raw;d=e.device;n=e.num_envs;c=e.graph_control;ik=e.cartesian_ik
 assert env.task=='branch'
 path=Path(e.hand.cfg.spawn.asset_path);tree=ET.parse(path).getroot()
 names=list(e.hand.joint_names);J=len(names);ref=e.hand.data.joint_pos[:1].clone()
 joints=[];children=set()
 for j in tree.findall('joint'):
  origin=j.find('origin');xyz=[float(x) for x in origin.get('xyz','0 0 0').split()] if origin is not None else [0]*3
  rpy=[float(x) for x in origin.get('rpy','0 0 0').split()] if origin is not None else [0]*3
  mat=trimesh.transformations.euler_matrix(*rpy);mat[:3,3]=xyz
  ax=j.find('axis');axis=torch.tensor([float(x) for x in ax.get('xyz','1 0 0').split()] if ax is not None else [1.,0.,0.],device=d)
  parent=j.find('parent').get('link');child=j.find('child').get('link');children.add(child)
  joints.append((parent,child,torch.tensor(mat,device=d,dtype=torch.float32),names.index(j.get('name')) if j.get('name') in names else None,axis,j.get('type')))
 root=next(l.get('name') for l in tree.findall('link') if l.get('name') not in children)
 ordered=[];available={root}
 while joints:
  ready=[j for j in joints if j[0] in available]
  if not ready:raise RuntimeError('Disconnected URDF')
  for j in ready:ordered.append(j);available.add(j[1]);joints.remove(j)
 def fk(q):
  count=len(q);T={root:torch.eye(4,device=d).expand(count,-1,-1)}
  for parent,child,origin,index,axis,kind in ordered:
   motion=torch.eye(4,device=d).repeat(count,1,1)
   if index is not None:
    if kind=='prismatic':motion[:,:3,3]=q[:,index,None]*axis
    else:
     x,y,z=axis;zero=x*0;K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
     angle=q[:,index,None,None]
     motion[:,:3,:3]=torch.eye(3,device=d)+angle.sin()*K+(1-angle.cos())*(K@K)
   T[child]=T[parent]@origin@motion
  return T
 transforms=fk(ref);anchor=next(name for name in e.hand.body_names if name in transforms)
 idx=e.hand.body_names.index(anchor);actual=torch.eye(4,device=d);actual[:3,:3]=matrix_from_quat(e.hand.data.body_quat_w[0,idx]);actual[:3,3]=e.hand.data.body_pos_w[0,idx]
 world=actual@torch.linalg.inv(transforms[anchor][0])
 offsets=[];radius=[];pad_reach=[];pad_names=[]
 for digit in range(1,6):
  link=next(l for l in tree.findall('link') if l.get('name').startswith(f'link_{digit}_') and any('finger_pad' in x.get('name','') for x in l.findall('collision')))
  pad_names.append(link.get('name'));pad=next(x for x in link.findall('collision') if 'finger_pad' in x.get('name',''))
  mesh=trimesh.load(str(path.parent/pad.find('geometry/mesh').get('filename')),force='mesh')
  bounds=mesh.bounds;offsets.append(torch.tensor(bounds.mean(0),device=d,dtype=torch.float32));radius.append(float((bounds[1]-bounds[0])[:2].min()/2));pad_reach.append(float(((bounds[1]-bounds[0])**2).sum()**.5/2))
 offsets=torch.stack(offsets);rad=torch.tensor(radius,device=d)
 def points(q,off,point_names=None):
  ts=fk(q);return torch.stack([(world@ts[name])[:,:3,3]+((world@ts[name])[:,:3,:3]@off[i,:,None]).squeeze(-1) for i,name in enumerate(pad_names if point_names is None else point_names)],1)
 predicted=points(ref,ik.tip_offsets,ik.finger_names);measured=ik.fingertip_positions_world()[:1];error=float((predicted-measured).norm(dim=-1).max())
 assert error<.001,f'FK validation mismatch {error} m'
 R=matrix_from_quat(e.branch.data.root_quat_w[0]);center=e.branch.data.root_pos_w[0];branch_radius=float(e.cfg.branch_radius)
 height=float(getattr(e.cfg,'branch_length',.18))
 limits=e.hand.root_physx_view.get_dof_limits()[0].to(d);lower=limits[:,0];upper=limits[:,1]
 # Match the actual action controller bounds for its controlled joints.
 lower[c.indices]=c.lower;upper[c.indices]=c.upper
 grid=torch.cartesian_prod(torch.linspace(-.03,.03,7,device=d),torch.linspace(-.03,.03,7,device=d))
 local_shift=torch.cat((grid,torch.zeros(len(grid),1,device=d)),-1)
 world_shift=(local_shift@R.T).repeat_interleave(16,0)
 count=len(world_shift);torch.manual_seed(20281028)
 q=(lower+(upper-lower)*torch.rand(count,J,device=d)).requires_grad_()
 with torch.no_grad():q[:16]=ref.expand(16,-1)
 angles=(torch.arange(count,device=d)%16)*(2*math.pi/16)
 directions=torch.stack([angles.cos(),angles.sin()],-1)[:,None,:].repeat(1,5,1);directions[:,1:]*=-1
 targets=directions*(branch_radius+rad)[None,:,None]
 relaxed=os.environ.get('BRANCH_REACH_RELAXED')=='1'
 angle_variable=angles.clone().requires_grad_();offset_angles=(torch.rand(count,4,device=d)*2-1)*1.396
 offset_angles.requires_grad_();center_radius=(branch_radius+rad)[None].repeat(count,1).requires_grad_()
 def get_targets():
  if not relaxed:return targets
  aa=angle_variable[:,None]+torch.cat((torch.zeros(count,1,device=d),math.pi+offset_angles),-1)
  return torch.stack((aa.cos(),aa.sin()),-1)*center_radius[:,:,None]
 optimizer=torch.optim.Adam([{'params':[q],'lr':.045},{'params':[angle_variable,offset_angles],'lr':.025},{'params':[center_radius],'lr':.0003}] if relaxed else [q],lr=.045)
 for iteration in range(400):
  p=(points(q,offsets)-center-world_shift[:,None,:])@R
  err=(p[:,:,:2]-get_targets()).square().sum(-1)+(p[:,:,2].abs()-(height/2-.008)).clamp_min(0).square()
  loss=(err[:,0]+err[:,1:].topk(2,largest=False).values.sum(-1)+.05*err[:,1:].sum(-1)).mean()
  optimizer.zero_grad();loss.backward();optimizer.step()
  with torch.no_grad():
   q.clamp_(lower,upper);offset_angles.clamp_(-1.396,1.396)
   center_radius.clamp_(min=branch_radius);center_radius.copy_(torch.minimum(center_radius,branch_radius+torch.tensor(pad_reach,device=d)))
  if iteration%40==0:save_json(out/'audit_progress.json',dict(phase='FK search',iteration=iteration,updated=time.time(),loss=float(loss)))
 with torch.no_grad():
  p=(points(q,offsets)-center-world_shift[:,None,:])@R;distance=(p[:,:,:2]-get_targets()).norm(dim=-1)
  score=distance[:,0]+distance[:,1:].topk(2,largest=False).values.sum(-1)
  candidates=(distance[:,0]<.003)&((distance[:,1:]<.003).sum(-1)>=2)&((p[:,:,2].abs()<height/2).sum(-1)>=3)
  first=torch.arange(len(grid),device=d)*16+score.reshape(len(grid),16).argmin(-1)
  remaining=score.argsort();remaining=remaining[~torch.isin(remaining,first)]
  chosen=torch.cat((first,remaining[:n-len(first)]));goal=q[chosen].detach();selected_shift=world_shift[chosen]
  assert len(chosen)==n
 save_json(out/'kinematic_search.json',dict(fk_validation_error_m=error,searches=count,iterations=400,chosen_world_shift_m=selected_shift.tolist(),original_branch_world_m=center.tolist(),candidate_count=int(candidates.sum()),best_finger_radial_errors_m=distance[chosen].tolist(),best_pad_positions_branch_local_m=p[chosen].tolist(),joint_names=names,chosen_joint_positions=goal.tolist(),relaxed_opposition=relaxed,scope='Necessary geometric feasibility only; no collision/force constraints in FK. Physical replay below retains all original collisions.',branch_radius_m=branch_radius,pad_halfwidth_m=radius))
 # Translate only at reset; never move the branch during an episode.
 from types import MethodType
 old_reset=e._reset_idx
 def shifted_reset(raw,ids):
  old_reset(ids);idx=slice(None) if ids is None else ids
  state=raw.branch.data.root_state_w[idx].clone();state[:,:3]+=selected_shift[idx]
  raw.branch.write_root_state_to_sim(state,ids)
  raw._compute_intermediate_values()
  p,r=raw._branch_relative_pose();raw.tactile_prev_pos[idx]=p[idx];raw.tactile_prev_quat[idx]=r[idx]
 e._reset_idx=MethodType(shifted_reset,e)
 # Replay through ordinary bounded actions; do not change hand geometry.
 rows=[];original=e._get_rewards;active={}
 def capture():
  r=original()
  if active:
   a=active;t=e.whole_hand_tactile;f=t.digit_forces();live=a['alive'];contact=(f[:,0]>=e.cfg.branch_contact_force_threshold)&((f[:,1:]>=e.cfg.branch_contact_force_threshold).sum(-1)>=2)
   a['contact']|=live&contact;a['opposed']|=live&contact&(t.opposition_cosine()<0);a['success']|=live&env.event
   a['hold']=torch.maximum(a['hold'],torch.where(live,e.tactile_streak,0));a['force']=max(a['force'],float(f[live].max()) if live.any() else 0.)
   if a['step']<120:a['initial_force']=torch.maximum(a['initial_force'],f.amax(-1))
   # Continue a successful diagnostic episode to measure sustained holding.
   e.reset_terminated.zero_();e.reset_buf=e.reset_time_outs.clone()
  return r
 e._get_rewards=capture
 for duration in [360,900]:
  active.clear();env.reset_seed=args.eval_seed;obs=env.reset();start=e.hand.data.joint_pos.clone();z=torch.zeros(n,device=d,dtype=torch.bool)
  active.update(alive=~z,contact=z.clone(),opposed=z.clone(),success=z.clone(),hold=z.long(),force=0.,initial_force=z.float(),step=0)
  for step in range(e.max_episode_length):
   active['step']=step
   target=start+(goal-start)*max(0.,min(1.,(step-119)/duration))
   val=target.index_select(-1,c.indices).clamp(c.lower,c.upper)
   action=2*(val-c.lower)/(c.upper-c.lower)-1
   obs,reward,done,info=env.step(action);active['alive']&=~done
   if step%120==0:save_json(out/'audit_progress.json',dict(phase='physical replay',ramp_steps=duration,step=step,updated=time.time()))
   if not active['alive'].any():break
  a=active;rows.append(dict(ramp_steps=duration,episodes=n,simultaneous=int(a['contact'].sum()),opposed=int(a['opposed'].sum()),successes=int(a['success'].sum()),max_hold_steps=int(a['hold'].max()),peak_digit_force_N=a['force'],initial_force_N=a['initial_force'].tolist(),per_candidate_hold_steps=a['hold'].tolist(),per_candidate_success=a['success'].tolist(),per_candidate_opposed=a['opposed'].tolist(),valid_sustained=((a['hold']>=180)&(a['initial_force']<1.)).tolist()))
  save_json(out/'physical_replay.json',rows)
 save_json(out/'completed.json',dict(status='completed',diagnostic=True,kinematic_candidates=int(candidates.sum()),physical_trials=rows))
