"""Every-substep key travel monitor; does not alter dynamics."""
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
