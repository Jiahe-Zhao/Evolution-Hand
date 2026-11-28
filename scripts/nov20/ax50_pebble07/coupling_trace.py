
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
import numpy as np
@wp.kernel
def collect(c:wp.array(dtype=int),sh:wp.array(dtype=int),sb:wp.array(dtype=int),item:int,f:wp.array(dtype=wp.vec3),n:wp.array(dtype=wp.vec3),dep:wp.array(dtype=float),ke:wp.array(dtype=float),kd:wp.array(dtype=float),mu:wp.array(dtype=float),cur:wp.array(dtype=int),out:wp.array2d(dtype=float)):
 i=wp.tid();j=cur[0]
 if i<c[0] and sb[sh[i]]==item and j<out.shape[0]:
  fn=wp.max(0.,wp.dot(f[i],n[i]));el=wp.max(0.,dep[i])*ke[i]
  wp.atomic_add(out,j,0,fn);wp.atomic_add(out,j,1,el);wp.atomic_add(out,j,2,wp.max(0.,fn-el))
  if fn>1.e-8:
   wp.atomic_add(out,j,3,1.);wp.atomic_max(out,j,4,wp.max(0.,dep[i])*1000.);wp.atomic_add(out,j,5,mu[i]);wp.atomic_add(out,j,6,kd[i])
@wp.kernel
def bones(cur:wp.array(dtype=int),out:wp.array2d(dtype=float),af:wp.array(dtype=wp.vec3),bi:wp.array(dtype=int),body:int,axis:wp.vec3,feedback:wp.array(dtype=wp.spatial_vector),bq:wp.array(dtype=wp.transform),bv:wp.array(dtype=wp.spatial_vector)):
 j=cur[0]
 if j<out.shape[0]:
  f=wp.vec3(0.)
  for i in range(bi.shape[0]):
   if bi[i]==body:f-=af[i]
  out[j,7]=wp.dot(f,axis);out[j,8]=wp.dot(wp.spatial_top(feedback[body]),axis);out[j,9]=wp.dot(wp.spatial_top(bv[body]),axis)
 cur[0]+=1

def install(h,axis):
 h.subtrace=wp.zeros((30000,10),dtype=float);h.subcursor=wp.zeros(1,dtype=int)
 body=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
 old=h.step
 def step(dt):
  old(dt);c=h.contacts
  wp.launch(collect,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,h.model.shape_body,h.item,h.out_f,c.soft_contact_normal,h.out_depth,h.vbd.body_particle_contact_penalty_k,h.vbd.body_particle_contact_material_kd,h.vbd.body_particle_contact_material_mu,h.subcursor,h.subtrace])
  wp.launch(bones,1,inputs=[h.subcursor,h.subtrace,h.anchor_force,h.bi,body,wp.vec3(*axis),h.feedback,h.state.body_q,h.state.body_qd])
 h.step=step;h._advance_graph=None
