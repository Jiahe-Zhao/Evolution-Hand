"""Diagnostic only: freeze rigid poses during the hold, preserving soft physics."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def freeze(flag:wp.array(dtype=int),fixed:wp.array(dtype=wp.transform),q:wp.array(dtype=wp.transform),qd:wp.array(dtype=wp.spatial_vector),count:int):
 i=wp.tid()
 if i<count and flag[0]!=0:q[i]=fixed[i];qd[i]=wp.spatial_vector()
def install(h):
 h.freeze_flag=wp.zeros(1,dtype=int);h.freeze_pose=wp.clone(h.state.body_q)
 old=h.rigid.step
 def step(s,n,c,contacts,dt):
  old(s,n,c,contacts,dt)
  wp.launch(freeze,h.hand_body_count,inputs=[h.freeze_flag,h.freeze_pose,n.body_q,n.body_qd,h.hand_body_count])
 h.rigid.step=step;h._advance_graph=None
