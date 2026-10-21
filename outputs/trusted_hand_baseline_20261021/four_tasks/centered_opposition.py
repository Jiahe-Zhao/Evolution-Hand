"""Pad-face opposition planner adapter and position/orientation teacher."""
import json,os
from pathlib import Path
import numpy as np
import torch
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from strike_opposition_rollout import contact

def place(e,out):
 from scipy.spatial.transform import Rotation
 from plan_centered_pad_faces import Model
 base=Path(__file__).parents[1];plans=json.loads((base/'centered_plans_inward.json').read_text())
 feasible=[r for r in plans if not r['intersections']]
 if not feasible:raise RuntimeError('No collision-free centered pad plan')
 row=min(feasible,key=lambda r:r['cost']);m=Model(base/'candidate/right/urdf/current_agent.urdf');R=np.array(row['rotation']);G=R.T;c=np.array(row['center']);newcenter=e.cone.data.root_pos_w[0].cpu().numpy().copy()
 root=e.hand.data.root_state_w.clone();root[0,:3]=torch.tensor(newcenter+G@(m.root[:3,3]-c),device=e.device);rq=Rotation.from_matrix(G@m.root[:3,:3]).as_quat();root[0,3:7]=torch.tensor(rq[[3,0,1,2]],device=e.device);root[:,7:]=0
 q=e.hand.data.joint_pos.clone()
 for name,value in row['q'].items():q[:,e.hand.joint_names.index(name)]=value
 q[:,e.driven_wrist_ids]=0
 e.hand.write_root_state_to_sim(root);e.hand.data.default_root_state[:]=root;e.hand.write_joint_state_to_sim(q,torch.zeros_like(q));e.hand.set_joint_position_target(q);e.prev_targets[:]=q;e.cur_targets[:]=q;e.graph_wrist_offset.zero_()
 state=e.cone.data.root_state_w.clone();state[:,3:7]=torch.tensor([1.,0.,0.,0.],device=e.device);state[:,7:]=0;e.cone.write_root_state_to_sim(state)
 support=e.tool_support.data.root_state_w.clone();support[:,:3]=state[:,:3];support[:,2]-=float(e.cfg.Cone_cfg.spawn.size[2]/2+.010);support[:,7:]=0;e.tool_support.write_root_state_to_sim(support)
 ts=m.fk(np.array([row['q'][n] for n in m.names]));offsets=[];normals=[]
 for i,(_,padname,p,n) in enumerate(m.pad_specs):
  parent=e.cartesian_ik.finger_names[i];T=np.linalg.inv(ts[parent])@ts[padname];offsets.append(T[:3,:3]@p+T[:3,3]);normals.append(T[:3,:3]@n)
 e.centered_pad_offsets=torch.tensor(np.array(offsets),device=e.device,dtype=torch.float32);e.centered_pad_normals=torch.tensor(np.array(normals),device=e.device,dtype=torch.float32)
 e.scene.write_data_to_sim();e.sim.forward();e.scene.update(0.)
 row.update(hand_root=root[0].cpu().tolist(),tool_world=state[0].cpu().tolist(),pad_offsets=offsets,normal_local_body=normals)
 (out/'centered_placement.json').write_text(json.dumps(row,default=lambda x:x.tolist(),indent=2))
 e.centered_plan=row

class Teacher:
 def __init__(self,e,graph,task):
  self.e=e;self.graph=graph;self.c=e.graph_control;self.task=task;self.stage='align';self.wrist=e.graph_wrist_offset.clone();self.leaf=0;self.tick=0;self.stable=0;self.hold=0;self.free_hold_pass=False;self.anchor=None;self.grasp_q=None
  p,n=self.pads();obj=e.cone;rot=obj.data.root_quat_w[:,None,:].expand(-1,5,-1);self.local=quat_apply_inverse(rot,p-obj.data.root_pos_w[:,None,:]);self.local[:,0,1:]=self.local[:,1:,1:].mean(1)
  self.depth=torch.zeros((1,5),device=e.device);self.sign=torch.tensor([[1.,-1.,-1.,-1.,-1.]],device=e.device);self.start=obj.data.root_pos_w.clone();self.initial_height=float(self.start[0,2]);self.initial_root=e.hand.data.body_pos_w[:,e.driven_hand_body_id].clone();self.trace=[]
 def pads(self):
  e=self.e;ids=e.cartesian_ik.body_ids;r=e.hand.data.body_quat_w[:,ids]
  p=e.hand.data.body_pos_w[:,ids]+quat_apply(r,e.centered_pad_offsets[None]);n=quat_apply(r,e.centered_pad_normals[None]);return p,n
 def action(self):
  e=self.e;c=self.c;e._compute_intermediate_values();p,n=self.pads();obj=e.cone;f=contact(e,'strike');q=e.hand.data.joint_pos.clone();rot=obj.data.root_quat_w[:,None,:].expand(-1,5,-1);self.tick+=1
  local=self.local.clone();desired_force=torch.tensor([[.8,.2,.2,.2,.2]],device=e.device)
  self.depth+=(desired_force-f).clamp(-.5,.5)*.00002;self.depth.clamp_(0,.0012)
  local[:,:,0]=self.sign*(float(e.cfg.Cone_cfg.spawn.size[0]/2)-self.depth)
  targets=obj.data.root_pos_w[:,None,:]+quat_apply(rot,local);outward=torch.zeros_like(local);outward[:,:,0]=self.sign;desired_normal=-quat_apply(rot,outward)
  self.tip_goal=targets;error=targets-p;angle_error=torch.cross(n,desired_normal,dim=-1)
  jacobians=e.hand.root_physx_view.get_jacobians();ik=e.cartesian_ik
  for i,(body,ids) in enumerate(zip(ik.body_ids,ik.joint_ids)):
   J=jacobians[:,body-1,:,ids];offset=quat_apply(e.hand.data.body_quat_w[:,body],e.centered_pad_offsets[i][None]-e.hand.data.body_com_pos_b[:,body]);jp=J[:,:3,:]+torch.cross(J[:,3:,:].transpose(1,2),offset[:,None,:].expand(-1,len(ids),-1),dim=-1).transpose(1,2)
   err=error[:,i].clone();err*=torch.clamp(.00025/err.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1)
   j=torch.cat([jp,J[:,3:,:]*.01],dim=1);rhs=torch.cat([err,angle_error[:,i]*.0003],dim=1);eye=torch.eye(len(ids),device=e.device)[None];dq=torch.linalg.solve(j.transpose(1,2)@j+eye*.00001,j.transpose(1,2)@rhs[:,:,None]).squeeze(-1);q[:,ids]+=dq.clamp(-.008,.008)
  centered=float((quat_apply_inverse(rot,p-obj.data.root_pos_w[:,None,:])[0,0,1:]-quat_apply_inverse(rot,p-obj.data.root_pos_w[:,None,:])[0,1:,1:].mean(0)).norm());alignment=(n*desired_normal).sum(-1)
  both=bool(f[0,0]>.08) and bool((f[0,1:]>.03).sum()>=1)
  aligned=centered<.006 and float(alignment.min())>.70
  if self.stage=='align':
   self.stable=self.stable+1 if both and aligned else max(0,self.stable-1)
   if self.stable>=60:self.stage='lift';self.grasp_q=q.clone();self.anchor=obj.data.root_pos_w.clone()
  if self.stage in ('lift','hold'):
   q=self.grasp_q.clone();self.wrist[:,2]=.012
   rise=float(obj.data.root_pos_w[0,2]-self.anchor[0,2])
   if rise>.007:self.stage='hold';self.hold+=1
   else:self.hold=0
   if self.hold>=180:self.free_hold_pass=True
  self.trace.append({'time_s':self.tick*e.step_dt,'stage':self.stage,'pad_positions':p[0].cpu().tolist(),'pad_normals':n[0].cpu().tolist(),'center_offset_m':centered,'normal_alignment':alignment[0].cpu().tolist(),'contact_N':f[0].cpu().tolist(),'free_hold_pass':self.free_hold_pass})
  desired=q.index_select(-1,c.indices).clamp(c.lower,c.upper);a=torch.zeros_like(e.graph_actions);a[:,:len(self.graph.joints)]=2*(desired-c.lower)/(c.upper-c.lower)-1;a[:,len(self.graph.joints):]=(self.wrist/e.wrist_action_scale).clamp(-1,1);return a
