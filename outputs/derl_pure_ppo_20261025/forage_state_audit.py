"""Read-only comparison of public rigid state and direct PhysX state."""
import torch,json
class StateAudit:
 def __init__(self,raw,out):
  self.r=raw;self.out=out;self.previous=None;self.steps=0;self.events=[];self.max_speed=0.;self.max_fd=0.;self.max_cache_error=0.;self.bad_frames=0;self.severe_streak=torch.zeros(raw.num_envs,dtype=torch.long,device=raw.device)
 def sample(self,alive):
  r=self.r;assets=(r.leaf_one,r.leaf_two)
  pos=torch.stack([x.data.root_pos_w for x in assets],1).clone();vel=torch.stack([x.data.root_lin_vel_w for x in assets],1).clone()
  direct_pos=torch.stack([x.root_physx_view.get_transforms()[:,:3] for x in assets],1).clone();direct_vel=torch.stack([x.root_physx_view.get_velocities()[:,:3] for x in assets],1).clone()
  error=torch.maximum((pos-direct_pos).abs().amax(-1),(vel-direct_vel).abs().amax(-1));speed=vel.norm(dim=-1)
  fd=torch.zeros_like(speed) if self.previous is None else ((pos-self.previous)/r.step_dt).norm(dim=-1)
  self.previous=pos;self.steps+=1
  if alive.any():
   self.max_speed=max(self.max_speed,float(speed[alive].max()));self.max_fd=max(self.max_fd,float(fd[alive].max()));self.max_cache_error=max(self.max_cache_error,float(error[alive].max()))
  severe=((speed>10)&(fd<1)).any(-1)&alive
  self.severe_streak=torch.where(severe,self.severe_streak+1,0)
  bad=(speed.amax(-1)>2)&alive;self.bad_frames+=int(bad.sum())
  if bad.any() and len(self.events)<20:
   i=int(bad.nonzero()[0]);self.events.append(dict(step=self.steps,env=i,speed=speed[i].tolist(),finite_difference_speed=fd[i].tolist(),direct_speed=direct_vel[i].norm(dim=-1).tolist(),cache_error=error[i].tolist(),position=(pos[i]-r.scene.env_origins[i]).tolist()))
  if self.steps%120==0 or bad.any():self.save()
 def save(self):
  data=dict(steps=self.steps,max_speed=self.max_speed,max_finite_difference_speed=self.max_fd,max_cache_error=self.max_cache_error,over_2m_s_env_frames=self.bad_frames,events=self.events)
  p=self.out/'state_audit.tmp';p.write_text(json.dumps(data,indent=2));p.replace(self.out/'state_audit.json')

 def guard(self):
  if int(self.severe_streak.max())>=3:
   self.save()
   raise RuntimeError('Forage PhysX velocity/position inconsistency persisted for 3 steps; preserving diagnostics, aborting contaminated run')
