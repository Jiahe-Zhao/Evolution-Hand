"""250ms recorded-target controlled diagnostic; zero velocities, not full task acceptance."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from damped_adapter import sync_ghosts
from contact_guard_geometry import install,install_controller
import csv,os
mode=sys.argv[1];dt=float(sys.argv[2]);frame=220
c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text());z=np.load(OUT/'shape_topwide_v1.npz')
h=PebbleRuntime(side='left',wrist='six_axis',position=np.asarray(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),object_xml=Path(__file__).parent/'assets/object_topwide_v1.xml')
from pebble_contact_tuning import set_hand_stone_tau
tuning=set_hand_stone_tau(h,.002)
install(h)
search=json.loads((Path(__file__).parent/'candidates/contact_search_v5.json').read_text())
install_controller(h,search)
for state in [h.state,h.next]:
 state.body_q.assign(z['body_q'][frame]);state.joint_q.assign(z['joint_q'][frame]);state.body_qd.zero_();state.joint_qd.zero_();state.particle_qd.zero_();p=state.particle_q.numpy();p[:h.real_count]=z['particles'][frame];state.particle_q.assign(p)
 wp.launch(sync_ghosts,len(h.pi),inputs=[state.particle_q,state.particle_qd,state.body_q,state.body_qd,h.model.body_com,h.gi,h.bi,h.local])
h.feedback.zero_();h.set_targets(z['targets'][frame])
caps=h.caps.numpy()
for key in ['thumb','index']:
 for name in search['fingers'][key]['names']:caps[h.action_indices[h.action_names.index(name)]]=.1 if mode=='baseline' else .02
h.caps.assign(caps)
wp.copy(h.guard_previous,h.state.joint_q)
h.guard_enabled.assign(np.array([int(mode=='guarded')],dtype=np.int32))
rows=[];depth_rows=[];contact_info=[];bodies=[h.state.body_q.numpy()];times=[0.]
sb=h.model.shape_body.numpy();tag=f'coupling_geometry_stiff_window_{mode}_{round(dt*1e6)}us'
for j in range(250):
 h.set_targets(z['targets'][frame+j//10+1])
 h.guard_depth_peak.zero_();h.guard_peak.zero_();h.guard_integral.zero_();h.advance(.001,dt=dt)
 depth_rows.append(h.guard_depth_peak.numpy().tolist())
 peak=h.guard_peak.numpy();mean=h.guard_integral.numpy()/.001;b=h.state.body_q.numpy();v=h.state.body_qd.numpy()[h.item]
 rows.append([h.time,*peak,*mean,*b[h.item,:3],*v]);bodies.append(b);times.append(h.time)
 n=int(h.contacts.soft_contact_count.numpy()[0]);ss=h.contacts.soft_contact_shape.numpy()[:n];sel=sb[ss]==h.item
 kd=h.vbd.body_particle_contact_material_kd.numpy()[:n][sel];ke=h.vbd.body_particle_contact_penalty_k.numpy()[:n][sel]
 contact_info.append(dict(time=h.time,target_soft_contacts=int(sel.sum()),kd_Ns_m=kd.tolist(),ke_N_m=ke.tolist(),target_soft_depth_mm=(h.out_depth.numpy()[:n][sel]*1000).tolist()))
a=np.array(rows);np.savetxt(OUT/(tag+'_trace.csv'),a,delimiter=',',header='time_s,thumb_peak_N,index_peak_N,soft_peak_N,soft_normal_peak_N,thumb_mean_N,index_mean_N,soft_mean_N,soft_normal_mean_N,x,y,z,v0,v1,v2,w0,w1,w2',comments='')
np.savez_compressed(OUT/(tag+'.npz'),body_q=bodies,time=times,body_names=z['body_names'])
r=dict(tuning=tuning,scope=__doc__,mode=mode,dt_s=dt,source_time_s=float(z['time'][frame]),object_xml=str(Path(__file__).parent/'assets/object_topwide_v1.xml'),finite=bool(np.isfinite(a).all()),max_native_depth_mm=(np.max(depth_rows,axis=0)*1000).tolist(),peak_forces_N=a[:,1:5].max(axis=0).tolist(),peak_linear_speed_m_s=float(np.linalg.norm(a[:,12:15],axis=1).max()),object_displacement_mm=((bodies[-1][h.item,:3]-bodies[0][h.item,:3])*1000).tolist(),contact_info=contact_info)
(OUT/(tag+'_result.json')).write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='contact_info'}),flush=True)
