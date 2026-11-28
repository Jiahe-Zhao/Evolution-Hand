
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
p=Path(__file__).parent;s=(p/'run_friction_monitor.py').read_text();s=s[:s.index("mode=os.environ.get('PROBE_MODE'")]
s=s.replace('h = PebbleRuntime(','''class PadRuntime(PebbleRuntime):
 def _add_brush(self,b):
  super()._add_brush(b)
  b.body_flags[self.item]=int(newton.BodyFlags.KINEMATIC)
h = PadRuntime(''')
s=s.replace("h.object_bodies, :3", "h.object_bodies, :3")
s+='''
from scipy.spatial.transform import Rotation as R
meta=json.loads(Path(os.environ['CAL_META']).read_text());which=meta['finger'];axis=np.array(meta['axis']);tangent=np.array(meta['tangent']);local=np.array(meta['local']);cp=np.load(OUT/'jitter_source_state_20261127.npz')
# Restore only hand; fixed specimen keeps its authored pose and free joint q.
for state in [h.state,h.next]:
 for key in ['body_q','body_qd','joint_q','joint_qd']:
  ar=getattr(state,key).numpy();old=cp[key];n=h.hand_body_count if key.startswith('body') else (h.hand_q_count if hasattr(h,'hand_q_count') and key=='joint_q' else h.hand_dof_count)
  if key=='joint_q':n=h.free_joint_rows[0]['q_start']
  ar[:n]=old[:n];getattr(state,key).assign(ar)
 for key in ['particle_q','particle_qd']:getattr(state,key).assign(cp[key])
h.feedback.zero_();h.guard_enabled.assign(np.array([0],dtype=np.int32))
target0=np.clip(cp['target'][h.action_indices]*h.signs,h.action_lower,h.action_upper);h.set_targets(target0)
bid=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/'+meta['body']));base_pose=cp['body_q'][bid];start=base_pose[:3]+R.from_quat(base_pose[3:]).apply(local);rot=R.from_quat(c['orientation']);period=.005;rows=[];bodies=[];particles=[];times=[];contacts=[];friction_samples=[];stop='complete'
for it in range(340):
 t=it*period;slide=0.
 if t<.1:press=0.;phase='settle'
 elif t<.6:press=.0007*smooth((t-.1)/.5);phase='press'
 elif t<.8:press=.0007;phase='hold'
 elif t<1.2:press=.0007;slide=.002*smooth((t-.8)/.4);phase='slide'
 elif t<1.3:press=.0007;slide=.002;phase='post_slide'
 else:press=.0007*(1-smooth((t-1.3)/.4));slide=.002;phase='unload'
 target=target0.copy();target[:3]+=rot.inv().apply(axis*press+tangent*slide);h.set_targets(target);h.guard_peak.zero_();h.guard_depth_peak.zero_();h.vector_integral.zero_();h.advance(period,dt=.000125)
 b=h.state.body_q.numpy();tip=b[bid,:3]+R.from_quat(b[bid,3:]).apply(local);delta=tip-start;v=h.vector_integral.numpy()/period;groups=[0] if which=='thumb' else [1,2];force=v[groups,:3].sum(0);normal=float(force@axis);tforce=float(force@tangent);cs=summarize(it);friction_samples.append(snapshot(h))
 row=dict(time_s=h.time,phase=phase,command_press_mm=press*1000,actual_tip_press_mm=float(delta@axis*1000),command_slide_mm=slide*1000,actual_tip_slide_mm=float(delta@tangent*1000),normal_N=normal,tangent_N=tforce,active_count=float(v[groups,9].sum()),rigid_depth_mm=float(h.guard_depth_peak.numpy()[0 if which=='thumb' else 1]*1000),soft_peak_N=float(h.guard_peak.numpy()[2]),other_hand_N=cs['other_hand_item_N'],specimen_move_mm=float(np.linalg.norm(b[h.item,:3]-np.array(meta['center']))*1000))
 rows.append(row);bodies.append(b);particles.append(h.state.particle_q.numpy()[:h.real_count]);times.append(h.time);contacts.append(cs)
 if it%40==0:print(row,flush=True)
 if not np.isfinite(b).all() or not np.isfinite(particles[-1]).all():stop='nonfinite';break
 if abs(normal)>5 or row['rigid_depth_mm']>.5:stop='force_or_penetration_stop';break
np.savez_compressed(OUT/f'{TAG}.npz',time=times,phase=[x['phase'] for x in rows],body_q=bodies,particles=particles,body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contacts,soft_samples=np.array(friction_samples,dtype=object))
with (OUT/f'{TAG}_trace.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=dict(finger=which,stop=stop,scope='fixed planar specimen, existing hand parameters; diagnostic not human validated',meta=meta,object_xml=str(object_xml),max_normal_N=max(x['normal_N'] for x in rows),max_specimen_move_mm=max(x['specimen_move_mm'] for x in rows),pair_info=h.pair_override_info)
(OUT/f'{TAG}_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
'''
(p/'run_pad_calibration.py').write_text(s)
