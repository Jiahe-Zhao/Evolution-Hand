from contact_audit_handle import *
from scipy.spatial.transform import Rotation as R
import csv,os
p=json.loads((OUT/os.environ.get('PLAN_FILE','open_support_plan.json')).read_text());rot=R.from_quat(p['orientation']);far=np.array([float(os.environ.get("FAR_X",".03")),0,0]);mount=np.array(p['position'])+far;tag=os.environ.get('RUN_TAG','open_support_v1');h=AuditedHandleRuntime(side='left',wrist='six_axis',position=mount,orientation=p['orientation'],gravity=(0,0,-9.81));pose=np.array(p['open_pose']);smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));T=float(os.environ.get('DURATION','2.3'));rows=[];contacts=[];bodies=[];particles=[];targets=[];stop='completed';sb=h.model.shape_body.numpy();times=[];joints=[]
meta=dict(tag=tag,action_names=h.action_names,action_indices=h.action_indices.tolist(),plan=p,mount=mount.tolist(),object_initial=h.state.body_q.numpy()[h.object_bodies].tolist(),object_mass_kg=h.model.body_mass.numpy()[h.object_bodies].tolist(),object_position_gains=h.model.joint_target_ke.numpy()[h.prop_dof_start:].tolist(),scope='open-hand support stage only, not complete grasp acceptance; actual Newton PD-controlled hand, free vessel and passive bail; no object force/pose drive')
(OUT/(tag+'_model.json')).write_text(json.dumps(meta,indent=2));fh=(OUT/(tag+'_trace.csv')).open('w');writer=csv.writer(fh);header=['time_s','phase','hinge_rad','lift_m','tilt_deg','body_contact_N','bail_contact_N','finite','max_soft_speed_m_s']+AUDIT_COLUMNS;writer.writerow(header)
for i in range(round(T/.01)):
 t=i*.01;target=np.zeros(len(h.action_names));q=np.zeros(16);delta=np.zeros(3)
 q=pose*smooth(t/.3);phase='open_hand_position'
 if t>=.3:delta=-far*smooth((t-.3)/.5);phase='open_hand_approach'
 if t>=.8:delta[2]+=.002*smooth((t-.8)/.2);phase='seat_pads_under_grip'
 if t>=1.:
  angle=np.deg2rad(75)*smooth((t-1.)/1.0);delta=-far+np.array([.064*(np.cos(angle)-1),0,.002+.064*np.sin(angle)]);phase='support_bail_open_hand' if t<2. else 'hold_bail_open_hand'
 target[:3]=rot.inv().apply(delta);target[6:22]=q;h.set_targets(target);h.advance(.01,dt=.000125 if t<.9 else float(os.environ.get('SIM_DT','.0000625')))
 bq=h.state.body_q.numpy();pq=h.state.particle_q.numpy();pv=h.state.particle_qd.numpy();qq=h.state.joint_q.numpy();finite=all(np.isfinite(x).all() for x in [bq,pq,pv,qq,h.state.joint_qd.numpy()]);loads=h.audit_load.numpy();tilt=np.rad2deg(np.arccos(np.clip(R.from_quat(bq[h.item,3:]).apply([0,0,1])[2],-1,1))) if finite else float('nan');row=[h.time,phase,float(qq[h.hinge_q]),float(bq[h.item,2]-h.z0),float(tilt),*loads.tolist(),finite,float(np.nanmax(np.linalg.norm(pv,axis=1))),*h.audit_values()];rows.append(row);writer.writerow(row);fh.flush();times.append(h.time);bodies.append(bq);joints.append(qq);particles.append(pq[:h.real_count]);targets.append(target)
 c=h.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);a=sb[c.rigid_contact_shape0.numpy()[:n]];b=sb[c.rigid_contact_shape1.numpy()[:n]];ff=c.force.numpy()[:n,:3];nn=c.rigid_contact_normal.numpy()[:n];p0=c.rigid_contact_point0.numpy()[:n];p1=c.rigid_contact_point1.numpy()[:n];sh0=c.rigid_contact_shape0.numpy()[:n];sh1=c.rigid_contact_shape1.numpy()[:n]
 for j in range(n):
  for own,other,sign in [(a[j],b[j],1),(b[j],a[j],-1)]:
   if 0<=own<h.hand_body_count and other in h.object_bodies:
    fn=abs(float(ff[j]@nn[j]))
    if fn>1e-8:contacts.append([h.time,'rigid',h.body_names[own],'item' if other==h.item else 'bail',*(sign*ff[j]).tolist(),fn,str(h.model.shape_label[sh1[j] if sign==1 else sh0[j]]).rsplit('/',1)[-1],*(p0[j] if sign==1 else p1[j]).tolist()])
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);body=sb[c.soft_contact_shape.numpy()[:n]];ff=h.out_f.numpy()[:n];nn=c.soft_contact_normal.numpy()[:n];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n]
 for j in np.flatnonzero(np.isin(body,h.object_bodies)):
  fn=abs(float(ff[j]@nn[j]))
  if fn>1e-8:
   pid=int(ii[j,np.argmax(bb[j])]);name=h.region_names[h.regions[pid]] if 0<=pid<h.real_count and h.regions[pid]>=0 else 'other';contacts.append([h.time,'soft',name,'item' if body[j]==h.item else 'bail',*ff[j].tolist(),fn,'soft_contact',float('nan'),float('nan'),float('nan')])
 if i%10==0:print('RUN',round(h.time,3),phase,'hinge',round(np.rad2deg(qq[h.hinge_q]),2),'lift',round(row[3]*1000,2),'tilt',round(tilt,2),'loads',loads.tolist(),'audit',h.audit_values(),flush=True)
 if i==219:np.savez_compressed(OUT/(tag+'_checkpoint.npz'),joint_q=h.state.joint_q.numpy(),joint_qd=h.state.joint_qd.numpy(),body_q=h.state.body_q.numpy(),body_qd=h.state.body_qd.numpy(),particle_q=h.state.particle_q.numpy(),particle_qd=h.state.particle_qd.numpy(),feedback=h.feedback.numpy(),target=h.target.numpy(),time=h.time)
 if i%10==0:np.savez_compressed(OUT/(tag+'_live.npz'),time=times,body_q=bodies,body_names=h.body_names_all,particles=particles,tri=h.model.tri_indices.numpy(),targets=targets,joint_q=joints)
 if not finite or row[8]>20 or h.audit_bad.numpy()[0]:stop='nonfinite_runaway_or_body_contact';break
fh.close();np.savez_compressed(OUT/(tag+'.npz'),time=times,body_q=bodies,body_names=h.body_names_all,particles=particles,tri=h.model.tri_indices.numpy(),targets=targets,joint_q=joints)
with (OUT/(tag+'_contacts.csv')).open('w') as f:w=csv.writer(f);w.writerow(['time_s','type','hand_part','object','fx_N','fy_N','fz_N','normal_N','object_shape','hand_local_x_m','hand_local_y_m','hand_local_z_m']);w.writerows(contacts)
(OUT/(tag+'_status.json')).write_text(json.dumps(dict(stop=stop,time=h.time),indent=2));print('DONE',stop,flush=True)
