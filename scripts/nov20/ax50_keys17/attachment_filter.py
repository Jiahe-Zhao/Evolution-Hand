
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
@wp.kernel
def mark_attached(count:wp.array(dtype=int),indices:wp.array(dtype=wp.vec3i),bary:wp.array(dtype=wp.vec3),shapes:wp.array(dtype=int),shape_body:wp.array(dtype=int),owner:wp.array(dtype=int),sentinel:int,removed:wp.array(dtype=int)):
 i=wp.tid()
 if i<count[0]:
  v=indices[i];b=shape_body[shapes[i]];match=True;active=False
  for k in range(3):
   if bary[i][k]>1.e-6:
    active=True
    if v[k]<0 or v[k]>=owner.shape[0]:match=False
    elif owner[v[k]]!=b:match=False
  if active and match and b>=0:
   indices[i]=wp.vec3i(sentinel,sentinel,sentinel);wp.atomic_add(removed,0,1)
