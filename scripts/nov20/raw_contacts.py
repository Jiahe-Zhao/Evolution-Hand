
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
from mujoco_warp._src.types import vec5
@wp.kernel
def collect(step:int, ge:wp.array[wp.vec2i],dist:wp.array[float],world:wp.array[int],nat:wp.array[int],tool:wp.array[int],dim:wp.array[int],addr:wp.array2d[int],fric:wp.array[vec5],ef:wp.array2d[float],cone:int,count:wp.array2d[int],out:wp.array4d[float]):
 i=wp.tid()
 if i>=nat[0] or dist[i]>0.:return
 a=ge[i][0];b=ge[i][1]
 if tool[a]==0 and tool[b]==0:return
 w=world[i];k=wp.atomic_add(count,step,w,1)
 if k>=32:return
 out[step,w,k,0]=float(a);out[step,w,k,1]=float(b);out[step,w,k,2]=dist[i];out[step,w,k,3]=float(dim[i]);out[step,w,k,4]=float(addr[i,0])
 for h in range(5):out[step,w,k,5+h]=fric[i][h]
 n=dim[i]
 if cone==0 and n>1:n=2*(n-1)
 for h in range(10):
    if h<n and addr[i,0]>=0:
     idx=addr[i,h];out[step,w,k,10+h]=ef[w,idx]
