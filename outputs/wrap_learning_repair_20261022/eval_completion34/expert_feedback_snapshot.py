"""Physical feedback teacher. No object attachment or object-state writes."""
import numpy as np
import torch


class Expert:
 def __init__(self,hand,env,model,root0,rootq,limits,fids,wids,device,dt,rotate,unrotate):
  self.hand=hand;self.root0=root0;self.root_id=hand.body_names.index("link_0_0");self.env=env;self.limits=limits;self.fids=fids;self.wids=wids;self.dt=dt;self.rotate=rotate
  self.bodies=[hand.body_names.index(name) for _,name,_,_ in model.pad_specs]
  self.offsets=torch.tensor(np.array([p for _,_,p,_ in model.pad_specs]),device=device,dtype=torch.float32)
  self.normals=torch.tensor(np.array([n for _,_,_,n in model.pad_specs]),device=device,dtype=torch.float32)
  self.ids=[[hand.joint_names.index(n) for n in model.names if n.startswith(f'link_{d}_') or f'to_link_{d}_' in n] for d in range(1,6)]
  p,n=self.pads();self.reference_normals=n.clone()
  # Each environment has the same nominal tool center before randomization.
  from __main__ import obj
  self.reference_p=p.clone()
  self.depth=torch.zeros((env.num_envs,5),device=device)
  self.sign=torch.tensor([1,-1,-1,-1,-1],device=device)
  self.goals=torch.tensor([2,.5,.5,.5,.5],device=device)
  self.lift=torch.full((env.num_envs,),-1.,device=device)
  self.held=env.target.clone();self.age=env.age.clone();self.loss_time=torch.zeros(env.num_envs,device=device)

 def pads(self):
  h=self.hand;r=h.data.body_quat_w[:,self.bodies]
  return h.data.body_pos_w[:,self.bodies]+self.rotate(r,self.offsets[None].expand(len(r),-1,-1)),self.rotate(r,self.normals[None].expand(len(r),-1,-1))

 def action(self):
  h=self.hand;e=self.env;dt=self.dt
  reset=e.age<self.age;self.depth[reset]=0;self.lift[reset]=-1;self.loss_time[reset]=0;self.age=e.age.clone()
  p,n=self.pads();active=(e.age*dt>.75)&(self.lift<0)
  self.depth+=(self.goals-e.filtered).div(self.goals).clamp(-1,1)*.0006*dt*active[:,None]
  self.depth.clamp_(-.004,.010)
  # Preserve validated pad coordinates relative to actual wrist displacement.
  # The tool is never teleported, constrained, or attached to the hand.
  from __main__ import obj
  desired_p=self.reference_p+(h.data.body_pos_w[:,self.root_id]-self.root0)[:,None,:]
  desired_p[:,:,0]-=self.sign*self.depth
  Jall=h.root_physx_view.get_jacobians();desired=e.target.clone()
  for f,(body,ids) in enumerate(zip(self.bodies,self.ids)):
   J=Jall[:,body-1,:,ids]
   lever=self.rotate(h.data.body_quat_w[:,body],self.offsets[f][None]-h.data.body_com_pos_b[:,body])
   jp=J[:,:3,:]+torch.cross(J[:,3:,:].transpose(1,2),lever[:,None,:].expand(-1,len(ids),-1),dim=-1).transpose(1,2)
   matrix=torch.cat([jp,J[:,3:,:]*.003],dim=1)
   rhs=torch.cat([desired_p[:,f]-p[:,f],torch.cross(n[:,f],self.reference_normals[:,f],dim=-1)*.003],dim=1)
   H=matrix.transpose(1,2)@matrix+torch.eye(len(ids),device=p.device)[None]*1e-6
   b=(matrix.transpose(1,2)@rhs[:,:,None]).squeeze(-1)
   q=h.data.joint_pos[:,ids]
   lo=(self.limits[ids,0]-q).clamp_min(-.12);hi=(self.limits[ids,1]-q).clamp_max(.12)
   lo=torch.minimum(lo,hi-1e-8)
   x=torch.linalg.solve(H,b[:,:,None]).squeeze(-1).clamp(lo,hi)
   # Small box-constrained quadratic solve, batched over environments.
   for _ in range(8):
    grad=(H@x[:,:,None]).squeeze(-1)-b
    bound=((x<=lo+1e-7)&(grad>0))|((x>=hi-1e-7)&(grad<0))
    free=(~bound).float();fixed=x*bound
    A=H*free[:,:,None]*free[:,None,:]+torch.diag_embed(bound.float())
    r=(b-(H@fixed[:,:,None]).squeeze(-1))*free
    x=(torch.linalg.solve(A,r[:,:,None]).squeeze(-1)+fixed).clamp(lo,hi)
   desired[:,ids]=torch.where(active[:,None],q+x,desired[:,ids])
  desired=e.target+(desired-e.target).clamp(-.25*dt,.25*dt)
  ready=(e.stable>=.75)&(self.lift<0)
  self.lift[ready]=e.age[ready]*dt;self.held[ready]=desired[ready]
  elapsed=(e.age*dt-self.lift).clamp_min(0);u=(elapsed/3).clamp(0,1)
  wrist=torch.zeros_like(e.wrist);wrist[:,2]=.03*u**3*(10-15*u+6*u*u)*(self.lift>=0)
  desired=torch.where((self.lift>=0)[:,None],self.held,desired)
  normal=e.contact[:,:,0].abs();touch=(normal[:,0]>.1)&(normal[:,1:].sum(1)>.1)
  self.loss_time=torch.where((self.lift>=0)&(~touch),self.loss_time+dt,torch.zeros_like(self.loss_time))
  lost=self.loss_time>.25
  self.lift[lost]=-1;wrist[lost]=0;self.loss_time[lost]=0
  action=torch.cat([2*(desired[:,self.fids]-self.limits[self.fids,0])/(self.limits[self.fids,1]-self.limits[self.fids,0])-1,wrist/.25],1)
  return action.clamp(-1,1)
