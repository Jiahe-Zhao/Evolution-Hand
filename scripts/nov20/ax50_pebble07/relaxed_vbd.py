"""Instance-local nonlinear iteration relaxation; no material/contact filtering changes."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def relax(previous:wp.array(dtype=wp.vec3),current:wp.array(dtype=wp.vec3),factor:float):
 i=wp.tid();current[i]=previous[i]+factor*(current[i]-previous[i])
def install(h,factor=0.5):
 v=h.vbd;v.relax_previous=wp.clone(v.particle_displacements);v.relax_initializing=False
 old_init=v._initialize_particles;old_truncate=v._penetration_free_truncation
 def initialize(*args):
  v.relax_initializing=True;old_init(*args);v.relax_initializing=False
  wp.copy(v.relax_previous,v.particle_displacements)
 def truncate(*args):
  if not v.relax_initializing:wp.launch(relax,h.model.particle_count,inputs=[v.relax_previous,v.particle_displacements,float(factor)])
  old_truncate(*args)
  wp.copy(v.relax_previous,v.particle_displacements)
 v._initialize_particles=initialize;v._penetration_free_truncation=truncate;h._advance_graph=None
