from keys_runtime import *
import adapter as shared_adapter
@wp.kernel
def wrist_pd(q:wp.array(dtype=float),qd:wp.array(dtype=float),target:wp.array(dtype=float),caps:wp.array(dtype=float),idx:wp.array(dtype=int),force:wp.array(dtype=float)):
 i=wp.tid();j=idx[i];kp=float(4.);kd=float(.08);cap=caps[j]
 if j<3:kp=3000.;kd=35.;cap=30.
 elif j<6:kp=20.;kd=.5;cap=2.
 force[j]=wp.clamp(kp*(target[j]-q[j])-kd*qd[j],-cap,cap)
class WristKeysRuntime(KeysRuntime):
 def __init__(self,*args,**kwargs):
  shared_adapter.control_pd_subset=wrist_pd
  super().__init__(*args,**kwargs)
