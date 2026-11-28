
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def depth(count:wp.array(dtype=int),shape:wp.array(dtype=int),sb:wp.array(dtype=int),item:int,dep:wp.array(dtype=float),current:wp.array(dtype=float),peak:wp.array(dtype=float)):
 i=wp.tid()
 if i<count[0] and sb[shape[i]]==item:
  d=wp.max(0.,dep[i]);wp.atomic_max(current,1,d);wp.atomic_max(peak,0,d)
def install(h):
 h.soft_depth_peak=wp.zeros(1);old=h.step
 def step(dt):
  old(dt);c=h.contacts
  wp.launch(depth,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,h.model.shape_body,h.item,h.out_depth,h.guard_depth,h.soft_depth_peak])
 h.step=step;h._advance_graph=None
