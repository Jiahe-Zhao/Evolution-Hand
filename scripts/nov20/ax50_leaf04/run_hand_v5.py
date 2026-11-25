from leaf_runtime import *
import json,csv
plan=json.loads((OUT_DIR/'plan_oriented.json').read_text());tag=os.environ.get('RUN_TAG','hand_v1')
h=LeafRuntime(side='left',wrist='six_axis',position=plan['position'],orientation=plan['orientation'],gravity=(0,0,-9.81));(OUT_DIR/(tag+'_model.json')).write_text(json.dumps(h.leaf_info(),indent=2));poses=np.array(plan['poses']);base=np.r_[np.zeros(6),poses[0],np.zeros(5)];smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
rows=[];contacts=[];body=[];particle=[];targets=[];sb=h.model.shape_body.numpy();leafmap={b:k for k,b in enumerate(h.leaf_bodies)};stop='completed'
def stage(u,finger):
 s=np.clip(u,0,1)*3;k=min(int(s),2);r=smooth(s-k);return poses[k,finger*4:finger*4+4]*(1-r)+poses[k+1,finger*4:finger*4+4]*r
hold0=stage(1.,0);hold1=stage(1.,1)
for i in range(490):
 t=i*.01;target=base.copy()
 if t<.5:target*=smooth(t/.5);phase='preshape'
 elif t<.7:phase='settle'
 elif t<1.4:target[6:10]=stage(1.*(t-.7)/.7,0);phase='open_leaf0'
 elif t<1.6:target[6:10]=hold0;phase='hold_leaf0'
 elif t<2.3:target[6:10]=hold0;target[10:14]=stage(1.*(t-1.6)/.7,1);phase='open_leaf1'
 elif t<3.3:target[6:14]=np.r_[hold0,hold1];phase='hold_both'
 elif t<3.9:target[6:14]=np.r_[hold0,hold1]*(1-smooth((t-3.3)/.6))+poses[0,:8]*smooth((t-3.3)/.6);phase='withdraw'
 else:
  phase='return';target[1]=-.008*smooth((t-3.9)/.3)
 h.set_targets(target);h.advance(.01,dt=.000125 if t<.5 else .0000625)
 q=h.state.joint_q.numpy();qd=h.state.joint_qd.numpy();pq=h.state.particle_q.numpy();pv=h.state.particle_qd.numpy();bq=h.state.body_q.numpy();finite=all(np.isfinite(x).all() for x in [q,qd,pq,pv,bq]);load=np.zeros((2,3));c=h.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);a=sb[c.rigid_contact_shape0.numpy()[:n]];b=sb[c.rigid_contact_shape1.numpy()[:n]];ff=c.force.numpy()[:n,:3];norm=c.rigid_contact_normal.numpy()[:n]
 for j in range(n):
  for own,other,sign in [(a[j],b[j],1),(b[j],a[j],-1)]:
   if other in leafmap and 0<=own<h.hand_body_count:
    fn=abs(float(np.dot(ff[j],norm[j])))
    if fn>1.e-8:
     name=h.body_names[own];k=leafmap[other];col=0 if name=='link_2_3' else 1 if name=='link_3_3' else 2;load[k,col]+=fn;contacts.append([h.time,k,'rigid',name,*list(sign*ff[j]),fn])
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);sh=sb[c.soft_contact_shape.numpy()[:n]];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n];ff=h.out_f.numpy()[:n];norm=c.soft_contact_normal.numpy()[:n]
 for j in np.flatnonzero(np.isin(sh,h.leaf_bodies)):
  fn=abs(float(np.dot(ff[j],norm[j])))
  if fn>1.e-8:
   weights={}
   for pid,w in zip(ii[j],bb[j]):
    if w>0 and 0<=pid<h.real_count:
     reg=h.region_names[h.regions[pid]] if h.regions[pid]>=0 else 'other';weights[reg]=weights.get(reg,0.)+float(w)
   k=leafmap[sh[j]]
   for name,w in weights.items():
    load[k,0 if name=='index_distal' else 2]+=fn*w;contacts.append([h.time,k,'soft',name,*(ff[j]*w).tolist(),fn*w])
 load[load<.0005]=0.

 speed=float(np.nanmax(np.linalg.norm(pv,axis=1)));rows.append([h.time,phase,*q[h.leaf_q_np],*load.ravel(),finite,speed,float(np.nanmax(abs(qd)))]);body.append(bq);particle.append(pq[:h.real_count]);targets.append(target)
 if i in [69,159,239,329,489]:np.savez_compressed(OUT_DIR/(tag+f'_checkpoint_{i}.npz'),joint_q=q,joint_qd=qd,particle_q=pq,particle_qd=pv,body_q=bq,body_qd=h.state.body_qd.numpy(),time=h.time,target=target)
 if i%10==0:print('HAND',round(h.time,3),phase,'deg',np.rad2deg(q[h.leaf_q_np]).round(2),'loads',load.round(3).tolist(),'softv',round(speed,3),flush=True)
 if not finite or speed>20 or np.max(abs(q[h.leaf_q_np]))>2.:stop='nonfinite_or_runaway';break
np.savez_compressed(OUT_DIR/(tag+'.npz'),time=[r[0] for r in rows],body_q=body,body_names=h.body_names_all,particles=particle,tri=h.model.tri_indices.numpy(),targets=targets,leaf_q=np.array([r[2:4] for r in rows]))
for suffix,head,data in [('trace',['time_s','phase','leaf0_rad','leaf1_rad','leaf0_index_N','leaf0_middle_N','leaf0_other_N','leaf1_index_N','leaf1_middle_N','leaf1_other_N','finite','max_soft_speed_m_s','max_generalized_speed'],rows),('contacts',['time_s','leaf','type','body_or_region','fx_N','fy_N','fz_N','normal_N'],contacts)]:
 with (OUT_DIR/(tag+'_'+suffix+'.csv')).open('w') as f:w=csv.writer(f);w.writerow(head);w.writerows(data)
(OUT_DIR/(tag+'_status.json')).write_text(json.dumps(dict(stop=stop,time=h.time,not_task_acceptance=True),indent=2));print('DONE',stop,flush=True)
