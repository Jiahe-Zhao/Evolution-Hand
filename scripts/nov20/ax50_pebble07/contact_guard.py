"""Scene07 instrumentation: target-specific forces, no contact/feedback modification."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp

@wp.kernel
def rigid_load(count:wp.array(dtype=int),a:wp.array(dtype=int),b:wp.array(dtype=int),sb:wp.array(dtype=int),forces:wp.array(dtype=wp.spatial_vector),normals:wp.array(dtype=wp.vec3),item:int,thumb:int,index:int,current:wp.array(dtype=float)):
 i=wp.tid()
 if i<count[0]:
  ba=sb[a[i]];bb=sb[b[i]];other=int(-1)
  if ba==item:other=bb
  elif bb==item:other=ba
  fn=wp.abs(wp.dot(wp.spatial_top(forces[i]),normals[i]))
  if other==thumb:wp.atomic_add(current,0,fn)
  elif other==index:wp.atomic_add(current,1,fn)

@wp.kernel
def soft_load(count:wp.array(dtype=int),sh:wp.array(dtype=int),sb:wp.array(dtype=int),forces:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),indices:wp.array(dtype=wp.vec3i),bary:wp.array(dtype=wp.vec3),regions:wp.array(dtype=int),item:int,real_count:int,current:wp.array(dtype=float)):
 i=wp.tid()
 if i<count[0] and sb[sh[i]]==item:
  k=int(0)
  if bary[i][1]>bary[i][k]:k=1
  if bary[i][2]>bary[i][k]:k=2
  p=indices[i][k]
  if p>=0 and p<real_count:
   fn=wp.abs(wp.dot(forces[i],normals[i]))
   if regions[p]==3:wp.atomic_add(current,1,fn)
   wp.atomic_add(current,3,fn)

@wp.kernel
def record(current:wp.array(dtype=float),peak:wp.array(dtype=float),integral:wp.array(dtype=float),feedback:wp.array(dtype=wp.spatial_vector),item:int,dt:float):
 i=wp.tid();v=current[i]
 if i==2:v=wp.length(wp.spatial_top(feedback[item]))
 wp.atomic_max(peak,i,v);integral[i]+=v*dt

def install(h):
 thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'))
 index=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
 h.guard_current=wp.zeros(4);h.guard_peak=wp.zeros(4);h.guard_integral=wp.zeros(4)
 old=h.step
 def step(dt):
  old(dt);h.guard_current.zero_();r=h.rigid_observation;s=h.contacts
  wp.launch(rigid_load,r.rigid_contact_max,inputs=[r.rigid_contact_count,r.rigid_contact_shape0,r.rigid_contact_shape1,h.model.shape_body,r.force,r.rigid_contact_normal,h.item,thumb,index,h.guard_current])
  wp.launch(soft_load,s.soft_contact_max,inputs=[s.soft_contact_count,s.soft_contact_shape,h.model.shape_body,h.out_f,s.soft_contact_normal,s.soft_contact_indices,s.soft_contact_barycentric,h.region_ids,h.item,h.real_count,h.guard_current])
  wp.launch(record,4,inputs=[h.guard_current,h.guard_peak,h.guard_integral,h.feedback,h.item,dt])
 h.step=step;h._advance_graph=None

@wp.kernel
def guarded_target(q:wp.array(dtype=float),command:wp.array(dtype=float),target:wp.array(dtype=float),previous:wp.array(dtype=float),held:wp.array(dtype=float),latched:wp.array(dtype=int),finger:wp.array(dtype=int),direction:wp.array(dtype=float),load:wp.array(dtype=float),enabled:wp.array(dtype=int),dt:float):
 j=wp.tid();fi=finger[j]
 if fi>=0:
  goal=command[j]
  if enabled[0]!=0:
   f=load[fi]
   if f>=.15 and latched[j]==0:
    latched[j]=1;held[j]=q[j]
   if latched[j]!=0:
    if f>.4:held[j]-=direction[j]*.25*dt
    elif f<.06:held[j]+=direction[j]*.015*dt
    held[j]=wp.clamp(held[j],q[j]-.003,q[j]+.003)
    goal=held[j]
   goal=previous[j]+wp.clamp(goal-previous[j],-.25*dt,.25*dt)
  target[j]=goal;previous[j]=goal

def install_controller(h,search):
 import numpy as np
 n=len(h.target);finger=np.full(n,-1,dtype=np.int32);direction=np.zeros(n,dtype=np.float32)
 for fi,key in enumerate(['thumb','index']):
  v=np.asarray(search['fingers'][key]['joint_values']);ds=np.sign(v[-1]-v[0])
  for k,name in enumerate(search['fingers'][key]['names']):
   ai=h.action_names.index(name);j=h.action_indices[ai];finger[j]=fi;direction[j]=ds[k]*h.signs[ai]
 h.guard_finger=wp.array(finger,dtype=int);h.guard_direction=wp.array(direction)
 h.guard_command=wp.clone(h.target);h.guard_previous=wp.clone(h.target);h.guard_held=wp.clone(h.target)
 h.guard_latched=wp.zeros(n,dtype=int);h.guard_enabled=wp.zeros(1,dtype=int)
 set_old=h.set_targets
 def set_targets(target):
  set_old(target);wp.copy(h.guard_command,h.target)
 h.set_targets=set_targets
 old=h.step
 def step(dt):
  wp.launch(guarded_target,n,inputs=[h.state.joint_q,h.guard_command,h.target,h.guard_previous,h.guard_held,h.guard_latched,h.guard_finger,h.guard_direction,h.guard_current,h.guard_enabled,dt])
  old(dt)
 h.step=step;h._advance_graph=None
