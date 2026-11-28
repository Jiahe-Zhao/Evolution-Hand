"""Short continuation from measured final state; solver warm-start not restored."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from contact_guard_geometry_bounded import install,install_controller
from soft_coupling_probe import install as install_probe
from scipy.spatial.transform import Rotation
mode=sys.argv[1]
c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())
cp=np.load(OUT/'coupled_topwide_kd05_final_state.npz');search=json.loads((Path(__file__).parent/'candidates/contact_search_v5.json').read_text())
h=PebbleRuntime(side='left',wrist='six_axis',position=np.asarray(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),object_xml=Path(__file__).parent/'assets/object_topwide_v1.xml')
install(h);install_controller(h,search);install_probe(h,.5)
for state in [h.state,h.next]:
 for key in ['body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd']:getattr(state,key).assign(cp[key])
h.feedback.assign(cp['feedback'])
caps=h.caps.numpy()
for key in ['thumb','index']:
 for name in search['fingers'][key]['names']:caps[h.action_indices[h.action_names.index(name)]]=.02
h.caps.assign(caps)
target=cp['target'][h.action_indices]*h.signs
if mode=='pose_hold':target=cp['joint_q'][h.action_indices]*h.signs
violations=[dict(name=h.action_names[j],target=float(target[j]),lower=float(h.action_lower[j]),upper=float(h.action_upper[j])) for j in range(len(target)) if target[j]<h.action_lower[j]-1e-6 or target[j]>h.action_upper[j]+1e-6]
print('RESTORED_TARGET_LIMITS',violations,flush=True)
# Preserve the already-recorded native target without adding a new clamp to this diagnostic.
native=cp['target'].copy();native[h.action_indices]=target*h.signs
h.target.assign(native);wp.copy(h.guard_command,h.target);wp.copy(h.guard_previous,h.target);wp.copy(h.guard_held,h.target)
flags=(h.guard_finger.numpy()>=0).astype(np.int32);h.guard_latched.assign(flags)
h.guard_enabled.assign(np.array([int(mode in ['guarded','bounded'])],dtype=np.int32))
thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'));idx=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
rows=[];poses=[];times=[];dt=.000125
for i in range(120):
 h.guard_peak.zero_();h.guard_integral.zero_();h.guard_depth_peak.zero_();h.advance(.001,dt=dt)
 b=h.state.body_q.numpy();v=h.state.body_qd.numpy()[h.item,:3];p=b[h.item,:3];center=(b[thumb,:3]+b[idx,:3])/2
 relative=Rotation.from_quat(b[thumb,3:]).inv().apply(p-center)
 rows.append([h.time,*p,*relative,*v,*h.guard_peak.numpy(),*h.guard_integral.numpy()/.001,*h.guard_depth_peak.numpy()*1000])
 poses.append(b);times.append(h.time)
a=np.array(rows);tag='jitter_'+mode
np.savetxt(OUT/(tag+'_trace.csv'),a,delimiter=',',header='time_s,x,y,z,relative_x,relative_y,relative_z,vx,vy,vz,thumb_peak_N,index_peak_N,soft_peak_N,soft_normal_peak_N,thumb_mean_N,index_mean_N,soft_mean_N,soft_normal_mean_N,thumb_depth_mm,index_depth_mm',comments='')
np.savez_compressed(OUT/(tag+'.npz'),time=times,body_q=poses,body_names=h.model.body_label)
w=a[a[:,0]>=.02];delta=w[:,4:7]-w[0,4:7]
r=dict(restored_target_limit_violations=violations,scope=__doc__,mode=mode,source='coupled_topwide_kd05_final_state.npz',source_time_s=float(cp['time']),dt_s=dt,finite=bool(np.isfinite(a).all()),analysis_window_s=[.02,.12],relative_span_xyz_mm=(np.ptp(w[:,4:7],axis=0)*1000).tolist(),relative_max_displacement_mm=float(np.linalg.norm(delta,axis=1).max()*1000),vertical_velocity_sign_changes=int(np.sum(np.diff(np.sign(w[:,9]))!=0)),speed_peak_m_s=float(np.linalg.norm(w[:,7:10],axis=1).max()),final_target_limit_violations=int(np.sum((h.target.numpy()[h.action_indices]*h.signs<h.action_lower-1e-6)|(h.target.numpy()[h.action_indices]*h.signs>h.action_upper+1e-6))),force_peaks_N=w[:,10:14].max(0).tolist(),mean_normal_force_N=w[:,14:16].mean(0).tolist(),both_force_bins_fraction=float(np.mean((w[:,14]>.001)&(w[:,15]>.001))),max_native_depth_mm=w[:,18:20].max(0).tolist())
(OUT/(tag+'_result.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
