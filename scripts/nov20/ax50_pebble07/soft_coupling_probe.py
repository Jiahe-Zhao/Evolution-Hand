"""Measure elastic/damping normal forces; optional target-only damping experiment."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp

@wp.kernel
def set_target_kd(n:wp.array(dtype=int),shape:wp.array(dtype=int),sb:wp.array(dtype=int),item:int,kd:float,values:wp.array(dtype=float)):
 i=wp.tid()
 if i<n[0] and sb[shape[i]]==item:values[i]=kd

@wp.kernel
def components(n:wp.array(dtype=int),shape:wp.array(dtype=int),sb:wp.array(dtype=int),item:int,force:wp.array(dtype=wp.vec3),normal:wp.array(dtype=wp.vec3),depth:wp.array(dtype=float),ke:wp.array(dtype=float),kd:wp.array(dtype=float),out:wp.array(dtype=float)):
 i=wp.tid()
 if i<n[0] and sb[shape[i]]==item:
  fn=wp.max(0.,wp.dot(force[i],normal[i]));elastic=wp.max(0.,depth[i])*ke[i]
  wp.atomic_add(out,0,fn);wp.atomic_add(out,1,elastic);wp.atomic_add(out,2,wp.max(0.,fn-elastic))
  if fn>1.e-8:
   wp.atomic_add(out,3,1.);wp.atomic_add(out,4,kd[i]);wp.atomic_max(out,5,wp.max(0.,depth[i])*1000.)

@wp.kernel
def save_peak(current:wp.array(dtype=float),peak:wp.array(dtype=float)):
 if current[0]>peak[0]:
  for i in range(6):peak[i]=current[i]

def install(h,kd=None):
 if kd is not None:
  old_refresh=h.vbd._refresh_body_particle_contact_state
  def refresh(contacts,refresh):
   old_refresh(contacts,refresh)
   if contacts is not None:
    wp.launch(set_target_kd,contacts.soft_contact_max,inputs=[contacts.soft_contact_count,contacts.soft_contact_shape,h.model.shape_body,h.item,float(kd),h.vbd.body_particle_contact_material_kd])
  h.vbd._refresh_body_particle_contact_state=refresh
 h.force_components=wp.zeros(6);h.force_components_peak=wp.zeros(6);h.force_components_window=wp.zeros(6)
 old=h.step
 def step(dt):
  old(dt);c=h.contacts;h.force_components.zero_()
  wp.launch(components,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,h.model.shape_body,h.item,h.out_f,c.soft_contact_normal,h.out_depth,h.vbd.body_particle_contact_penalty_k,h.vbd.body_particle_contact_material_kd,h.force_components])
  wp.launch(save_peak,1,inputs=[h.force_components,h.force_components_peak]);wp.launch(save_peak,1,inputs=[h.force_components,h.force_components_window])
 h.step=step;h._advance_graph=None
