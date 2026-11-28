"""Every-substep key travel monitor; does not alter dynamics."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
from repair_keys_runtime import KeysRuntime
@wp.kernel
def monitor(q:wp.array[float],indices:wp.array[int],phase:wp.array[int],peaks:wp.array2d[float],bad:wp.array[int]):
 i=wp.tid();x=q[indices[i]]
 if not wp.isfinite(x):wp.atomic_add(bad,0,1)
 else:wp.atomic_max(peaks,phase[0],i,x)
class AuditedKeys(KeysRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.audit_indices=wp.array(self.key_q,dtype=int);self.audit_phase=wp.zeros(1,dtype=int);self.audit_peaks=wp.zeros((12,4),dtype=float);self.audit_bad=wp.zeros(1,dtype=int)
 def step(self,dt=.000125):
  super().step(dt)
  wp.launch(monitor,4,inputs=[self.state.joint_q,self.audit_indices,self.audit_phase,self.audit_peaks,self.audit_bad])
