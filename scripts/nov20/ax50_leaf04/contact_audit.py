"""Per-substep contact audit, without altering dynamics or contact forces."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from leaf_runtime import *
@wp.kernel
def rigid_load(count:wp.array[int],s0:wp.array[int],s1:wp.array[int],sb:wp.array[int],roles:wp.array[int],leaves:wp.array[int],force:wp.array[wp.spatial_vector],normal:wp.array[wp.vec3],out:wp.array[float]):
 i=wp.tid()
 if i>=count[0]:return
 a=sb[s0[i]];b=sb[s1[i]];leaf=-1;role=-1
 if a>=0 and b>=0:
  if leaves[a]>=0 and roles[b]>=0:leaf=leaves[a];role=roles[b]
  elif leaves[b]>=0 and roles[a]>=0:leaf=leaves[b];role=roles[a]
 if leaf>=0:
  f=wp.vec3(force[i][0],force[i][1],force[i][2]);wp.atomic_add(out,leaf*3+role,wp.abs(wp.dot(f,normal[i])))
@wp.kernel
def soft_load(count:wp.array[int],shape:wp.array[int],sb:wp.array[int],leaves:wp.array[int],indices:wp.array[wp.vec3i],bary:wp.array[wp.vec3],regions:wp.array[int],real_count:int,index_region:int,force:wp.array[wp.vec3],normal:wp.array[wp.vec3],out:wp.array[float]):
 i=wp.tid()
 if i>=count[0]:return
 body=sb[shape[i]]
 if body<0:return
 leaf=leaves[body]
 if leaf<0:return
 fn=wp.abs(wp.dot(force[i],normal[i]))
 for j in range(3):
  p=indices[i][j];w=bary[i][j]
  if p>=0 and p<real_count and w>0.:
   role=2
   if regions[p]==index_region:role=0
   wp.atomic_add(out,leaf*3+role,fn*w)
@wp.kernel
def tally(load:wp.array[float],counts:wp.array[int],mins:wp.array[float],maxs:wp.array[float]):
 counts[0]+=1
 other=load[1]+load[2]+load[3]+load[5]
 if load[0]>.01 and load[4]>.01 and other<.0005:counts[1]+=1
 mins[0]=wp.min(mins[0],load[0]);mins[1]=wp.min(mins[1],load[4]);maxs[0]=wp.max(maxs[0],other);maxs[1]=wp.max(maxs[1],load[0]);maxs[2]=wp.max(maxs[2],load[4])
class AuditedLeafRuntime(LeafRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  roles=np.full(self.model.body_count,-1,dtype=np.int32)
  for i,name in enumerate(self.body_names):roles[i]=0 if name=='link_2_3' else 1 if name=='link_3_3' else 2
  leaves=np.full(self.model.body_count,-1,dtype=np.int32)
  for k,i in enumerate(self.leaf_bodies):leaves[i]=k
  self.audit_roles=wp.array(roles,dtype=int);self.audit_leaves=wp.array(leaves,dtype=int);self.audit_regions=wp.array(np.asarray(self.regions,dtype=np.int32),dtype=int);self.audit_index_region=self.region_names.index('index_distal');self.audit_load=wp.zeros(6);self.audit_counts=wp.zeros(2,dtype=int);self.audit_mins=wp.zeros(2);self.audit_maxs=wp.zeros(3)
 def step(self,dt=.00025):
  super().step(dt)
  if not hasattr(self,'audit_load'):return
  self.audit_load.zero_();c=self.rigid_observation
  wp.launch(rigid_load,c.rigid_contact_max,inputs=[c.rigid_contact_count,c.rigid_contact_shape0,c.rigid_contact_shape1,self.model.shape_body,self.audit_roles,self.audit_leaves,c.force,c.rigid_contact_normal,self.audit_load])
  c=self.contacts
  wp.launch(soft_load,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,self.model.shape_body,self.audit_leaves,c.soft_contact_indices,c.soft_contact_barycentric,self.audit_regions,self.real_count,self.audit_index_region,self.out_f,c.soft_contact_normal,self.audit_load])
  wp.launch(tally,1,inputs=[self.audit_load,self.audit_counts,self.audit_mins,self.audit_maxs])
 def advance(self,duration=.01,dt=.00025):
  self.audit_counts.zero_();self.audit_mins.fill_(1.e10);self.audit_maxs.zero_();super().advance(duration,dt)
 def audit_values(self):return [*self.audit_counts.numpy().tolist(),*self.audit_mins.numpy().tolist(),*self.audit_maxs.numpy().tolist()]
AUDIT_COLUMNS=['substeps','qualified_substeps','min_leaf0_index_N','min_leaf1_middle_N','max_unassigned_N','max_leaf0_index_N','max_leaf1_middle_N']
