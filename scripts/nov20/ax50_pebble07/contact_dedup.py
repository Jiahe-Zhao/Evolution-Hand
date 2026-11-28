"""Merge coincident endpoint contact records, target shape only; retain distinct normals."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def choose(count:wp.array(dtype=int),ii:wp.array(dtype=wp.vec3i),bb:wp.array(dtype=wp.vec3),ss:wp.array(dtype=int),sb:wp.array(dtype=int),item:int,np:int,keys:wp.array(dtype=int)):
 i=wp.tid()
 if i<count[0] and sb[ss[i]]==item:
  for k in range(3):
   if ii[i][k]>=0 and bb[i][k]>.99999:wp.atomic_min(keys,ss[i]*np+ii[i][k],i)
@wp.kernel
def compact(count:wp.array(dtype=int),ii:wp.array(dtype=wp.vec3i),pp:wp.array(dtype=int),bb:wp.array(dtype=wp.vec3),ss:wp.array(dtype=int),xp:wp.array(dtype=wp.vec3),xv:wp.array(dtype=wp.vec3),nn:wp.array(dtype=wp.vec3),sb:wp.array(dtype=int),item:int,np:int,keys:wp.array(dtype=int),oc:wp.array(dtype=int),oi:wp.array(dtype=wp.vec3i),op:wp.array(dtype=int),ob:wp.array(dtype=wp.vec3),os:wp.array(dtype=int),oxp:wp.array(dtype=wp.vec3),oxv:wp.array(dtype=wp.vec3),on:wp.array(dtype=wp.vec3),removed:wp.array(dtype=int)):
 i=wp.tid()
 if i<count[0]:
  keep=True
  if sb[ss[i]]==item:
   for k in range(3):
    if ii[i][k]>=0 and bb[i][k]>.99999:
     winner=keys[ss[i]*np+ii[i][k]]
     if winner!=i and wp.length(xp[i]-xp[winner])<1.e-6 and wp.dot(nn[i],nn[winner])>.99999:keep=False
  if keep:
   j=wp.atomic_add(oc,0,1);oi[j]=ii[i];op[j]=pp[i];ob[j]=bb[i];os[j]=ss[i];oxp[j]=xp[i];oxv[j]=xv[i];on[j]=nn[i]
  else:wp.atomic_add(removed,0,1)
def install(h):
 c=h.contacts;dst=h.collision.contacts();keys=wp.full(h.model.shape_count*h.model.particle_count,2147483647,dtype=int);h.dedup_removed=wp.zeros(1,dtype=int)
 old=h.rigid.step
 def step(s,n,control,contacts,dt):
  keys.fill_(2147483647);dst.soft_contact_count.zero_()
  wp.launch(choose,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_indices,c.soft_contact_barycentric,c.soft_contact_shape,h.model.shape_body,h.item,h.model.particle_count,keys])
  wp.launch(compact,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_indices,c.soft_contact_particle,c.soft_contact_barycentric,c.soft_contact_shape,c.soft_contact_body_pos,c.soft_contact_body_vel,c.soft_contact_normal,h.model.shape_body,h.item,h.model.particle_count,keys,dst.soft_contact_count,dst.soft_contact_indices,dst.soft_contact_particle,dst.soft_contact_barycentric,dst.soft_contact_shape,dst.soft_contact_body_pos,dst.soft_contact_body_vel,dst.soft_contact_normal,h.dedup_removed])
  for field in ['count','indices','particle','barycentric','shape','body_pos','body_vel','normal']:wp.copy(getattr(c,'soft_contact_'+field),getattr(dst,'soft_contact_'+field))
  old(s,n,control,contacts,dt)
 h.rigid.step=step;h._advance_graph=None
