
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from contact_audit_handle import *
from scipy.spatial.transform import Rotation as R
import csv,os
p=json.loads((OUT/'plan.json').read_text());rot=R.from_quat(p['orientation']);far=np.array([float(os.environ.get("FAR_X",".09")),0,0]);mount=np.array(p['position'])+far;tag=os.environ.get('RUN_TAG','handle_v1');h=AuditedHandleRuntime(side='left',wrist='six_axis',position=mount,orientation=p['orientation'],gravity=(0,0,-9.81));pose=np.array(p['hook_pose']);smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));T=float(os.environ.get('DURATION','4.3'));rows=[];contacts=[];bodies=[];particles=[];targets=[];stop='completed';sb=h.model.shape_body.numpy();times=[]
meta=dict(tag=tag,action_names=h.action_names,action_indices=h.action_indices.tolist(),plan=p,mount=mount.tolist(),object_initial=h.state.body_q.numpy()[h.object_bodies].tolist(),object_mass_kg=h.model.body_mass.numpy()[h.object_bodies].tolist(),object_position_gains=h.model.joint_target_ke.numpy()[h.prop_dof_start:].tolist(),scope='actual Newton PD-controlled hand, free vessel and passive bail; no object force/pose drive')
(OUT/(tag+'_model.json')).write_text(json.dumps(meta,indent=2));fh=(OUT/(tag+'_trace.csv')).open('w');writer=csv.writer(fh);header=['time_s','phase','hinge_rad','lift_m','tilt_deg','body_contact_N','bail_contact_N','finite','max_soft_speed_m_s']+AUDIT_COLUMNS;writer.writerow(header)
start=0
source=os.environ.get('RESUME_TAG','')
if source:
 cp=np.load(OUT/(source+'_checkpoint.npz'))
 for state in [h.state,h.next]:
  for key in ['joint_q','joint_qd','body_q','body_qd','particle_q','particle_qd']:getattr(state,key).assign(cp[key])
 h.feedback.assign(cp['feedback']);h.target.assign(cp['target']);h.time=float(cp['time']);start=round(h.time/.01)
 prev=np.load(OUT/(source+'.npz'));times=list(prev['time'][:start]);bodies=list(prev['body_q'][:start]);particles=list(prev['particles'][:start]);targets=list(prev['targets'][:start])
 with (OUT/(source+'_trace.csv')).open() as f:rows=list(csv.reader(f))[1:start+1]
 writer.writerows(rows);fh.flush()
 with (OUT/(source+'_contacts.csv')).open() as f:contacts=[r for r in list(csv.reader(f))[1:] if float(r[0])<=h.time+1e-7]
 meta['checkpoint_continuation']=dict(source=source,time_s=h.time,scope='whole-world recorded state restored, no selective object repositioning; diagnostic continuation')
 (OUT/(tag+'_model.json')).write_text(json.dumps(meta,indent=2))
for i in range(start,round(T/.01)):
 t=i*.01;target=np.zeros(len(h.action_names));q=np.zeros(16);delta=np.zeros(3)
 if t<.4:q[:4]=pose[:4]*smooth(t/.4);q[12:]=pose[12:]*smooth(t/.4);phase='fold_other_fingers'
 else:
  q[:4]=pose[:4];q[12:]=pose[12:];q[4:12]=pose[4:12]*smooth((t-.4)/.5);phase='form_hook'
 if t>=1.:delta=-far*smooth((t-1.)/.6);phase='approach_under_bail'
 if t>=1.7:
  u=smooth((t-1.7)/1.4);a=np.pi/2*u;delta=-far+np.array([.064*(np.cos(a)-1),0,.064*np.sin(a)+.003*smooth((t-1.7)/.3)]);phase='rotate_and_lift' if t<3.3 else 'hold'
 if t>=3.1:
  delta[2]+=-.045*smooth((t-3.18)/.1)+.080*smooth((t-3.38)/.35)
  delta[0]+=.020*smooth((t-3.1)/.08)-.042*smooth((t-3.28)/.1)
  phase='reposition_below_grip' if t<3.38 else ('lift_from_below' if t<3.73 else 'hold')
 target[:3]=rot.inv().apply(delta);target[6:22]=q;h.set_targets(target);h.advance(.01,dt=.000125 if t<.9 else float(os.environ.get('SIM_DT','.0000625')))
 bq=h.state.body_q.numpy();pq=h.state.particle_q.numpy();pv=h.state.particle_qd.numpy();qq=h.state.joint_q.numpy();finite=all(np.isfinite(x).all() for x in [bq,pq,pv,qq,h.state.joint_qd.numpy()]);loads=h.audit_load.numpy();tilt=np.rad2deg(np.arccos(np.clip(R.from_quat(bq[h.item,3:]).apply([0,0,1])[2],-1,1))) if finite else float('nan');row=[h.time,phase,float(qq[h.hinge_q]),float(bq[h.item,2]-h.z0),float(tilt),*loads.tolist(),finite,float(np.nanmax(np.linalg.norm(pv,axis=1))),*h.audit_values()];rows.append(row);writer.writerow(row);fh.flush();times.append(h.time);bodies.append(bq);particles.append(pq[:h.real_count]);targets.append(target)
 c=h.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);a=sb[c.rigid_contact_shape0.numpy()[:n]];b=sb[c.rigid_contact_shape1.numpy()[:n]];ff=c.force.numpy()[:n,:3];nn=c.rigid_contact_normal.numpy()[:n]
 for j in range(n):
  for own,other,sign in [(a[j],b[j],1),(b[j],a[j],-1)]:
   if 0<=own<h.hand_body_count and other in h.object_bodies:
    fn=abs(float(ff[j]@nn[j]))
    if fn>1e-8:contacts.append([h.time,'rigid',h.body_names[own],'item' if other==h.item else 'bail',*(sign*ff[j]).tolist(),fn])
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);body=sb[c.soft_contact_shape.numpy()[:n]];ff=h.out_f.numpy()[:n];nn=c.soft_contact_normal.numpy()[:n];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n]
 for j in np.flatnonzero(np.isin(body,h.object_bodies)):
  fn=abs(float(ff[j]@nn[j]))
  if fn>1e-8:
   pid=int(ii[j,np.argmax(bb[j])]);name=h.region_names[h.regions[pid]] if 0<=pid<h.real_count and h.regions[pid]>=0 else 'other';contacts.append([h.time,'soft',name,'item' if body[j]==h.item else 'bail',*ff[j].tolist(),fn])
 if i%10==0:print('RUN',round(h.time,3),phase,'hinge',round(np.rad2deg(qq[h.hinge_q]),2),'lift',round(row[3]*1000,2),'tilt',round(tilt,2),'loads',loads.tolist(),'audit',h.audit_values(),flush=True)
 if i==309:np.savez_compressed(OUT/(tag+'_checkpoint.npz'),joint_q=h.state.joint_q.numpy(),joint_qd=h.state.joint_qd.numpy(),body_q=h.state.body_q.numpy(),body_qd=h.state.body_qd.numpy(),particle_q=h.state.particle_q.numpy(),particle_qd=h.state.particle_qd.numpy(),feedback=h.feedback.numpy(),target=h.target.numpy(),time=h.time)
 if i%10==0:np.savez_compressed(OUT/(tag+'_live.npz'),time=times,body_q=bodies,body_names=h.body_names_all,particles=particles,tri=h.model.tri_indices.numpy(),targets=targets)
 if not finite or row[8]>20 or h.audit_bad.numpy()[0]:stop='nonfinite_runaway_or_body_contact';break
fh.close();np.savez_compressed(OUT/(tag+'.npz'),time=times,body_q=bodies,body_names=h.body_names_all,particles=particles,tri=h.model.tri_indices.numpy(),targets=targets)
with (OUT/(tag+'_contacts.csv')).open('w') as f:w=csv.writer(f);w.writerow(['time_s','type','hand_part','object','fx_N','fy_N','fz_N','normal_N']);w.writerows(contacts)
(OUT/(tag+'_status.json')).write_text(json.dumps(dict(stop=stop,time=h.time),indent=2));print('DONE',stop,flush=True)
