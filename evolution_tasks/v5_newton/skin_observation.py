"""Named metacarpal/digit observations. Forces are per body, not per skin shape."""
import torch
from frame_math import inverse_rotate_xyzw
class SkinObservation:
 def __init__(self,hand,tactile):
  self.hand=hand;self.tactile=tactile
  self.names=[f'link_{d}_{s}' for d in range(1,6) for s in range(4)]
  self.ids=[hand.body_names.index(n) for n in self.names]
  self.root=hand.body_names.index('link_0_0')
  self.contract=dict(version='skin_regions_v1',regions=self.names,semantics=['metacarpal','proximal','middle','distal'],force_scope='body-target resultant; original skeleton and cover contacts combined',fields=['root_frame_position_m','root_frame_com_linear_velocity_m_s','body_frame_normal_force_N_per_target','body_frame_friction_force_N_per_target','contact_flag_per_target'],threshold_N=.001)
 def read(self):
  d=self.hand.data;pose=d.body_link_pose_w.torch;root=pose[:,self.root]
  if getattr(self,"inspect_raw",False):
   for name,x in (("body_link_pose_w",pose),("body_com_vel_w",d.body_com_vel_w.torch)):
    bad=~torch.isfinite(x)
    if bad.any():
     i,j,k=bad.nonzero()[0].tolist()
     raise FloatingPointError(f"raw={name} env={i} body={self.hand.body_names[j]} component={k} vector={x[i,j].tolist()}")
  p=inverse_rotate_xyzw(root[:,None,3:],pose[:,self.ids,:3]-root[:,None,:3])
  v=inverse_rotate_xyzw(root[:,None,3:],d.body_com_vel_w.torch[:,self.ids,:3]-d.body_com_vel_w.torch[:,self.root,None,:3])
  n,f=self.tactile.local_forces();n=n[:,self.ids];f=f[:,self.ids];contact=(n.norm(dim=-1)>.001).float()
  result=dict(position=p,velocity=v,normal=n,friction=f,contact=contact)
  for field,t in result.items():
   bad=~torch.isfinite(t)
   if bad.any():
    idx=bad.nonzero()[0].tolist();env,region=idx[:2]
    raw={name:value[env].tolist() for name,value in {
        "region_pose":pose[:,self.ids[region]],"root_pose":root,
        "region_twist":d.body_com_vel_w.torch[:,self.ids[region]],"root_twist":d.body_com_vel_w.torch[:,self.root]}.items()}
    raise FloatingPointError(f"skin field={field} index={idx} body={self.names[region]} raw={raw}")
  return result
 def vector(self):
  r=self.read()
  return torch.cat([r['position'].flatten(1)/.1,r['velocity'].flatten(1),r['normal'].flatten(1)/5,r['friction'].flatten(1)/5,r['contact'].flatten(1)],-1)
