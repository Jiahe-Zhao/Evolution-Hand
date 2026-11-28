"""Short recorded-target replay with reset velocities; diagnostic, not an exact checkpoint or task acceptance."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from damped_adapter import sync_ghosts
import csv
mode=sys.argv[1];c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text());z=np.load(OUT/'shape_topwide_v1.npz');frame=220
h=PebbleRuntime(side='left',wrist='six_axis',position=np.asarray(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),object_xml=Path(__file__).parent/'assets/object_topwide_v1.xml')
for state in [h.state,h.next]:
 state.body_q.assign(z['body_q'][frame]);state.joint_q.assign(z['joint_q'][frame]);state.body_qd.zero_();state.joint_qd.zero_();state.particle_qd.zero_();p=state.particle_q.numpy();p[:h.real_count]=z['particles'][frame];state.particle_q.assign(p)
 wp.launch(sync_ghosts,len(h.pi),inputs=[state.particle_q,state.particle_qd,state.body_q,state.body_qd,h.model.body_com,h.gi,h.bi,h.local])
h.feedback.zero_();caps=h.caps.numpy();search=json.loads((Path(__file__).parent/'candidates/contact_search_v5.json').read_text())
for name in search['fingers']['thumb']['names']+search['fingers']['index']['names']:caps[h.action_indices[h.action_names.index(name)]]=.1
h.caps.assign(caps)
sb=h.model.shape_body.numpy();rows=[];bodies=[z['body_q'][frame]];times=[float(z['time'][frame])];dt=.000125;tag='penetration_replay_'+mode
for j in range(25):
 h.set_targets(z['targets'][frame+j+1])
 for i in range(80):
  if mode=='no_target_soft_reaction':
   fb=h.feedback.numpy();fb[h.item]=0;h.feedback.assign(fb)
  h.step(dt)
  b=h.state.body_q.numpy();v=h.state.body_qd.numpy()[h.item];applied=h.next.body_f.numpy()[h.item,:3];feedback=h.feedback.numpy()[h.item,:3]
  r=h.rigid_observation;n=int(r.rigid_contact_count.numpy()[0]);a=sb[r.rigid_contact_shape0.numpy()[:n]];bb=sb[r.rigid_contact_shape1.numpy()[:n]];f=r.force.numpy()[:n,:3]
  force=f[a==h.item].sum(0)-f[bb==h.item].sum(0)
  rows.append(np.r_[float(z['time'][frame])+h.time,b[h.item,:3],v,applied,feedback,force])
 bodies.append(b);times.append(float(z['time'][frame])+h.time)
 print('WINDOW',mode,j+1,'position_mm',(b[h.item,:3]*1000).tolist(),'soft_feedback_N',float(np.linalg.norm(feedback)),flush=True)
a=np.array(rows);np.savetxt(OUT/(tag+'_trace.csv'),a,delimiter=',',header='time_s,x,y,z,v0,v1,v2,v3,v4,v5,applied_soft_fx,applied_soft_fy,applied_soft_fz,next_soft_fx,next_soft_fy,next_soft_fz,rigid_on_first_sign_fx,rigid_on_first_sign_fy,rigid_on_first_sign_fz',comments='')
np.savez_compressed(OUT/(tag+'.npz'),body_q=bodies,time=times,body_names=z['body_names'])
result=dict(scope=__doc__,mode=mode,object_xml=str(Path(__file__).parent/'assets/object_topwide_v1.xml'),source_time_s=float(z['time'][frame]),duration_s=h.time,dt_s=dt,samples=len(a),finite=bool(np.isfinite(a).all()),peak_applied_soft_force_N=float(np.max(np.linalg.norm(a[:,10:13],axis=1))),peak_generated_soft_force_N=float(np.max(np.linalg.norm(a[:,13:16],axis=1))),peak_linear_speed_m_s=float(np.max(np.linalg.norm(a[:,4:7],axis=1))),object_displacement_mm=((a[-1,1:4]-z['body_q'][frame,h.item,:3])*1000).tolist(),peak_soft_time_s=float(a[np.argmax(np.linalg.norm(a[:,10:13],axis=1)),0]))
(OUT/(tag+'_result.json')).write_text(json.dumps(result,indent=2));print('REPLAY_RESULT',json.dumps(result),flush=True)
