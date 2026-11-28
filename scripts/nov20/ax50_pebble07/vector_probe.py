
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def rigid(n:wp.array(dtype=int),a:wp.array(dtype=int),b:wp.array(dtype=int),sb:wp.array(dtype=int),f:wp.array(dtype=wp.spatial_vector),norm:wp.array(dtype=wp.vec3),item:int,thumb:int,index:int,hand:int,dt:float,out:wp.array2d(dtype=float)):
 i=wp.tid()
 if i<n[0]:
  ba=sb[a[i]];bb=sb[b[i]];other=int(-2);sg=float(0.)
  if ba==item:other=bb;sg=1.
  elif bb==item:other=ba;sg=-1.
  if sg!=0.:
   g=int(3)
   if other==thumb:g=0
   elif other==index:g=1
   elif other<0 or other>=hand:g=4
   v=wp.spatial_top(f[i])*sg;nn=norm[i];nf=nn*wp.dot(v,nn);tf=v-nf
   for k in range(3):
    wp.atomic_add(out,g,k,v[k]*dt);wp.atomic_add(out,g,k+3,nf[k]*dt);wp.atomic_add(out,g,k+6,tf[k]*dt)
   if wp.length(v)>.00001:wp.atomic_add(out,g,9,dt)
@wp.kernel
def soft(n:wp.array(dtype=int),sh:wp.array(dtype=int),sb:wp.array(dtype=int),f:wp.array(dtype=wp.vec3),norm:wp.array(dtype=wp.vec3),ii:wp.array(dtype=wp.vec3i),bary:wp.array(dtype=wp.vec3),regions:wp.array(dtype=int),real:int,item:int,dt:float,out:wp.array2d(dtype=float)):
 i=wp.tid()
 if i<n[0] and sb[sh[i]]==item:
  k=int(0)
  if bary[i][1]>bary[i][k]:k=1
  if bary[i][2]>bary[i][k]:k=2
  pid=ii[i][k];g=int(3)
  if pid>=0 and pid<real:
   if regions[pid]==3:g=2
  v=-f[i];nn=norm[i];nf=nn*wp.dot(v,nn);tf=v-nf
  for j in range(3):
   wp.atomic_add(out,g,j,v[j]*dt);wp.atomic_add(out,g,j+3,nf[j]*dt);wp.atomic_add(out,g,j+6,tf[j]*dt)
  if wp.length(v)>.00001:wp.atomic_add(out,g,9,dt)
def install(h):
 thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'));index=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'));h.vector_integral=wp.zeros((5,10));old=h.step
 def step(dt):
  old(dt);r=h.rigid_observation;s=h.contacts
  wp.launch(rigid,r.rigid_contact_max,inputs=[r.rigid_contact_count,r.rigid_contact_shape0,r.rigid_contact_shape1,h.model.shape_body,r.force,r.rigid_contact_normal,h.item,thumb,index,h.hand_body_count,dt,h.vector_integral])
  wp.launch(soft,s.soft_contact_max,inputs=[s.soft_contact_count,s.soft_contact_shape,h.model.shape_body,h.out_f,s.soft_contact_normal,s.soft_contact_indices,s.soft_contact_barycentric,h.region_ids,h.real_count,h.item,dt,h.vector_integral])
 h.step=step;h._advance_graph=None
