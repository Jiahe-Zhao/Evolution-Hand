"""Reset-only collision-free opposing placement; no attachment or rollout teleportation."""
import json, math
import numpy as np
import torch

def place(e,out,search=True):
 import fcl
 from scipy.spatial.transform import Rotation
 from imported_hand_geometry import hand_collision_objects
 tips=e.cartesian_ik.fingertip_positions_world()[0].cpu().numpy()
 bodies,_=hand_collision_objects(e,0)
 axis=tips[0,:2]-tips[1:,:2].mean(0);axis/=np.linalg.norm(axis)
 mid=(tips[0,:2]+tips[1:,:2].mean(0))/2
 old=e.cone.data.root_state_w.clone();size=np.array(e.cfg.Cone_cfg.spawn.size)
 candidates=[]
 # Align one box face with the thumb-to-finger opposition axis.
 yaw=math.atan2(axis[1],axis[0]);rot=Rotation.from_euler('z',yaw)
 for dx in np.linspace(-.018,.018,19):
  for dy in np.linspace(-.018,.018,19):
   p=old[0,:3].cpu().numpy().copy();p[:2]=mid+[dx,dy]
   signed=(tips[:,:2]-p[:2])@axis
   if signed[0]<size[0]/2+.001 or np.max(signed[1:])>-size[0]/2-.001:continue
   obj=fcl.CollisionObject(fcl.Box(*size),fcl.Transform(rot.as_matrix(),p))
   if any(fcl.collide(obj,b,fcl.CollisionRequest(),fcl.CollisionResult()) for _,b in bodies):continue
   cost=np.linalg.norm(p[:2]-mid)
   candidates.append((cost,p))
 if not candidates and search:
  original=e.hand.data.joint_pos.clone()
  for spread in [1.0,1.4,1.7,.3,0.]:
   for fraction in [.25,.4,.6,.75]:
    q=original.clone()
    limits=e.hand.root_physx_view.get_dof_limits().to(e.device)
    for j,name in enumerate(e.hand.joint_names):
     if name=='link_1_thumb_spread_joint':q[:,j]=spread
     elif 'to_link' in name:q[:,j]=limits[:,j,1]*fraction
    e.hand.write_joint_state_to_sim(q,torch.zeros_like(q));e.hand.set_joint_position_target(q)
    e.prev_targets[:]=q;e.cur_targets[:]=q
    e.scene.write_data_to_sim();e.sim.forward();e.scene.update(0.)
    if place(e,out,False):return True
  raise RuntimeError('No opposing reset posture found within original joint limits')
 if not candidates:
  (out/'opposition_placement.json').write_text(json.dumps({'feasible':False,'tips':tips.tolist(),'axis':axis.tolist()}))
  return False
 _,p=min(candidates,key=lambda x:x[0]);state=old.clone();state[0,:3]=torch.tensor(p,device=e.device);quat=rot.as_quat();state[0,3:7]=torch.tensor(quat[[3,0,1,2]],device=e.device);state[:,7:]=0
 e.cone.write_root_state_to_sim(state)
 support=e.tool_support.data.root_state_w.clone();support[:,:3]=state[:,:3];support[:,2]-=float(size[2]/2+.010);support[:,7:]=0;e.tool_support.write_root_state_to_sim(support)
 e.scene.write_data_to_sim();e.sim.forward();e.scene.update(0.)
 (out/'opposition_placement.json').write_text(json.dumps({'feasible':True,'position':p.tolist(),'axis':axis.tolist(),'signed_tip_distances_m':((tips[:,:2]-p[:2])@axis).tolist(),'candidate_count':len(candidates),'rollout_object_pose_writes':False},indent=2))

 return True
