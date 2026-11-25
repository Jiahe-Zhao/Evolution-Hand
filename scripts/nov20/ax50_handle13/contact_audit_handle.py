from handle_runtime import *
@wp.kernel
def rigid_load(count:wp.array[int],s0:wp.array[int],s1:wp.array[int],sb:wp.array[int],hand_count:int,item:int,bail:int,f:wp.array[wp.spatial_vector],n:wp.array[wp.vec3],out:wp.array[float]):
 i=wp.tid()
 if i>=count[0]:return
 a=sb[s0[i]];b=sb[s1[i]];obj=-1
 if a>=0 and a<hand_count:obj=b
 elif b>=0 and b<hand_count:obj=a
 k=-1
 if obj==item:k=0
 elif obj==bail:k=1
 if k>=0:wp.atomic_add(out,k,wp.abs(wp.dot(wp.vec3(f[i][0],f[i][1],f[i][2]),n[i])))
@wp.kernel
def soft_load(count:wp.array[int],shape:wp.array[int],sb:wp.array[int],item:int,bail:int,f:wp.array[wp.vec3],n:wp.array[wp.vec3],out:wp.array[float]):
 i=wp.tid()
 if i>=count[0]:return
 obj=sb[shape[i]];k=-1
 if obj==item:k=0
 elif obj==bail:k=1
 if k>=0:wp.atomic_add(out,k,wp.abs(wp.dot(f[i],n[i])))
@wp.kernel
def tally(load:wp.array[float],q:wp.array[float],bq:wp.array[wp.transform],item:int,hinge:int,z0:float,h0:float,counts:wp.array[int],stats:wp.array[float],bad:wp.array[int]):
 counts[0]+=1
 if load[0]>.0005:bad[0]=1
 lift=wp.transform_get_translation(bq[item])[2]-z0;up=wp.quat_rotate(wp.transform_get_rotation(bq[item]),wp.vec3(0.,0.,1.));tilt=wp.acos(wp.clamp(up[2],-1.,1.))*57.2957795;angle=(q[hinge]-h0)*57.2957795
 if angle>=45. and lift>=.025 and tilt<=15. and load[0]<=.0005 and load[1]>.01 and bad[0]==0:counts[1]+=1
 stats[0]=wp.max(stats[0],load[0]);stats[1]=wp.min(stats[1],load[1]);stats[2]=wp.min(stats[2],lift);stats[3]=wp.max(stats[3],tilt);stats[4]=wp.min(stats[4],angle);stats[5]=wp.max(stats[5],load[1])
class AuditedHandleRuntime(HandleRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.item,self.bail=self.object_bodies;self.z0=float(self.state.body_q.numpy()[self.item,2]);self.h0=float(self.state.joint_q.numpy()[self.hinge_q]);self.audit_load=wp.zeros(2);self.audit_counts=wp.zeros(2,dtype=int);self.audit_stats=wp.zeros(6);self.audit_bad=wp.zeros(1,dtype=int)
 def step(self,dt=.00025):
  super().step(dt)
  if not hasattr(self,'audit_load'):return
  self.audit_load.zero_();c=self.rigid_observation
  wp.launch(rigid_load,c.rigid_contact_max,inputs=[c.rigid_contact_count,c.rigid_contact_shape0,c.rigid_contact_shape1,self.model.shape_body,self.hand_body_count,self.item,self.bail,c.force,c.rigid_contact_normal,self.audit_load]);c=self.contacts
  wp.launch(soft_load,c.soft_contact_max,inputs=[c.soft_contact_count,c.soft_contact_shape,self.model.shape_body,self.item,self.bail,self.out_f,c.soft_contact_normal,self.audit_load])
  wp.launch(tally,1,inputs=[self.audit_load,self.state.joint_q,self.state.body_q,self.item,self.hinge_q,self.z0,self.h0,self.audit_counts,self.audit_stats,self.audit_bad])
 def advance(self,duration=.01,dt=.0000625):
  self.audit_counts.zero_();self.audit_stats.assign(np.array([0,1e10,1e10,0,1e10,0],dtype=np.float32));super().advance(duration,dt)
 def audit_values(self):return [*self.audit_counts.numpy().tolist(),*self.audit_stats.numpy().tolist(),int(self.audit_bad.numpy()[0])]
AUDIT_COLUMNS=['substeps','qualified_substeps','max_body_contact_N','min_bail_contact_N','min_lift_m','max_tilt_deg','min_hinge_deg','max_bail_contact_N','ever_body_contact']
