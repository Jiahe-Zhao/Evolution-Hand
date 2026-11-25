"""Short frozen-target relaxation, not a task rerun or exact checkpoint continuation."""
from pebble_runtime import *
from damped_adapter import sync_ghosts
import sys
@wp.kernel
def observe(rc:wp.array[int],a:wp.array[int],b:wp.array[int],rf:wp.array[wp.spatial_vector],rn:wp.array[wp.vec3],sc:wp.array[int],ss:wp.array[int],sf:wp.array[wp.vec3],sn:wp.array[wp.vec3],si:wp.array[wp.vec3i],bw:wp.array[wp.vec3],regions:wp.array[int],sb:wp.array[int],bq:wp.array[wp.transform],item:int,thumb:int,index:int,hand_count:int,row:int,out:wp.array2d[float]):
 t= float(0.);ix=float(0.);other=float(0.);support=float(0.)
 for k in range(rc[0]):
  ba=sb[a[k]];bb=sb[b[k]]
  if ba==item or bb==item:
   partner=bb if ba==item else ba;f=wp.abs(wp.dot(wp.spatial_top(rf[k]),rn[k]))
   if partner==thumb:t+=f
   elif partner==index:ix+=f
   elif partner>=0 and partner<hand_count:other+=f
   else:support+=f
 for k in range(sc[0]):
  if sb[ss[k]]==item:
   f=wp.abs(wp.dot(sf[k],sn[k]));weight=float(0.)
   for j in range(3):
    pid=si[k][j]
    if pid>=0 and pid<regions.shape[0]:
     if regions[pid]==3:weight+=bw[k][j]
   ix+=weight*f;other+=(1.-weight)*f
 pos=wp.transform_get_translation(bq[item]);out[row,0]=t;out[row,1]=ix;out[row,2]=other;out[row,3]=support;out[row,4]=pos[0];out[row,5]=pos[1];out[row,6]=pos[2]
case=sys.argv[1];dt=float(sys.argv[2]);iterations=int(sys.argv[3]);c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text());z=np.load(OUT/'pinch_trial_v6.npz',allow_pickle=True)
h=PebbleRuntime(side='left',wrist='six_axis',position=np.array(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),iterations=iterations)
for state in [h.state,h.next]:
 state.body_q.assign(z['body_q'][-2]);state.joint_q.assign(z['joint_q'][-2]);state.body_qd.zero_();state.joint_qd.zero_();state.particle_qd.zero_();p=state.particle_q.numpy();p[:h.real_count]=z['particles'][-2];state.particle_q.assign(p)
 wp.launch(sync_ghosts,len(h.pi),inputs=[state.particle_q,state.particle_qd,state.body_q,state.body_qd,h.model.body_com,h.gi,h.bi,h.local])
h.feedback.zero_();h.set_targets(z['targets'][-2]);n=round(.03/dt);out=wp.zeros((n,7));names=h.model.body_label;thumb=next(i for i,v in enumerate(names) if str(v).endswith('/link_1_2'));index=next(i for i,v in enumerate(names) if str(v).endswith('/link_2_3'))
for i in range(n):
 h.step(dt);r=h.rigid_observation;s=h.contacts
 wp.launch(observe,1,inputs=[r.rigid_contact_count,r.rigid_contact_shape0,r.rigid_contact_shape1,r.force,r.rigid_contact_normal,s.soft_contact_count,s.soft_contact_shape,h.out_f,s.soft_contact_normal,s.soft_contact_indices,s.soft_contact_barycentric,h.region_ids,h.model.shape_body,h.state.body_q,h.item,thumb,index,h.hand_body_count,i,out])
a=out.numpy();tail=a[round(.01/dt):];data=dict(scope='30 ms fixed-target relaxation of v6 recorded pose; velocities/feedback reset, not exact original trajectory',source_frame_time=float(z['time'][-2]),dt_s=dt,vbd_iterations=iterations,finite=bool(np.isfinite(a).all()),peak_N=a[:,:4].max(0).tolist(),tail_peak_N=tail[:,:4].max(0).tolist(),tail_mean_N=tail[:,:4].mean(0).tolist(),tail_std_N=tail[:,:4].std(0).tolist(),columns=['thumb','index','other_hand','support'],tail_dual_duty=float(np.mean((tail[:,0]>.001)&(tail[:,1]>.001))),target_displacement_mm=float(np.linalg.norm(a[-1,4:]-a[0,4:])*1000))
np.savez_compressed(OUT/(case+'.npz'),dt=dt,samples=a);(OUT/(case+'.json')).write_text(json.dumps(data,indent=2));print('PROBE_RESULT',json.dumps(data),flush=True)
