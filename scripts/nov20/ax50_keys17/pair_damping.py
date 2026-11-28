"""Pair-specific damping diagnostic. Retains collision, elasticity and friction."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os
import warp as wp
from newton.solvers import SolverVBD
@wp.kernel
def set_pair_damping(count:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],shape:wp.array[int],target:int,palm_count:int,kd:float,values:wp.array[float]):
 i=wp.tid()
 if i>=count[0] or shape[i]!=target:return
 is_palm=True
 for j in range(3):
  if bary[i][j]>1.e-5 and (indices[i][j]<0 or indices[i][j]>=palm_count):is_palm=False
 if is_palm:values[i]=kd
class PairVBD(SolverVBD):
 def _refresh_body_particle_contact_state(self,contacts,refresh):
  super()._refresh_body_particle_contact_state(contacts,refresh)
  if contacts is not None and hasattr(self,'pair_shape'):
   wp.launch(set_pair_damping,contacts.soft_contact_max,inputs=[contacts.soft_contact_count,contacts.soft_contact_indices,contacts.soft_contact_barycentric,contacts.soft_contact_shape,self.pair_shape,self.pair_palm_count,self.pair_kd,self.body_particle_contact_material_kd])

@wp.kernel
def set_key_damping(count:wp.array[int],shape:wp.array[int],is_key:wp.array[int],kd:float,values:wp.array[float]):
 i=wp.tid()
 if i<count[0] and is_key[shape[i]]!=0:values[i]=kd
class KeyPairVBD(PairVBD):
 def _refresh_body_particle_contact_state(self,contacts,refresh):
  super()._refresh_body_particle_contact_state(contacts,refresh)
  if contacts is not None and hasattr(self,'key_shape_flags'):
   wp.launch(set_key_damping,contacts.soft_contact_max,inputs=[contacts.soft_contact_count,contacts.soft_contact_shape,self.key_shape_flags,self.key_contact_kd,self.body_particle_contact_material_kd])
