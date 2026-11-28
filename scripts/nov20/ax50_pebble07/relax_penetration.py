"""20ms pose-relaxation diagnostic; velocities and solver warm start reset, not task replay."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from damped_adapter import sync_ghosts
from scipy.spatial.transform import Rotation
mode=sys.argv[1];frame=int(sys.argv[2]) if len(sys.argv)>2 else 230
c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text());z=np.load(OUT/'shape_topwide_v1.npz')
h=PebbleRuntime(side='left',wrist='six_axis',position=np.asarray(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),object_xml=Path(__file__).parent/'assets/object_topwide_v1.xml')
for state in [h.state,h.next]:
 state.body_q.assign(z['body_q'][frame]);state.joint_q.assign(z['joint_q'][frame]);state.body_qd.zero_();state.joint_qd.zero_();state.particle_qd.zero_();p=state.particle_q.numpy();p[:h.real_count]=z['particles'][frame];state.particle_q.assign(p)
 wp.launch(sync_ghosts,len(h.pi),inputs=[state.particle_q,state.particle_qd,state.body_q,state.body_qd,h.model.body_com,h.gi,h.bi,h.local])
default_caps=h.caps.numpy();h.feedback.zero_();h.set_targets(z['targets'][frame]);h.caps.zero_();h.control.joint_f.zero_()
if mode in ['full_hold','full_posehold','full_graph']:
 caps=default_caps.copy();search=json.loads((Path(__file__).parent/'candidates/contact_search_v5.json').read_text())
 for name in search['fingers']['thumb']['names']+search['fingers']['index']['names']:caps[h.action_indices[h.action_names.index(name)]]=.1
 h.caps.assign(caps)
 if mode=='full_posehold':h.set_targets(z['joint_q'][frame][h.action_indices])
thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'));sb=h.model.shape_body.numpy();labels=list(map(str,h.model.shape_label));rows=[]
def sample(t):
 b=h.state.body_q.numpy();rel=Rotation.from_quat(b[thumb,3:]).inv().apply(b[h.item,:3]-b[thumb,:3]);return dict(time_s=t,stone_world=b[h.item,:3].tolist(),thumb_world=b[thumb,:3].tolist(),stone_in_thumb=rel.tolist(),stone_velocity=h.state.body_qd.numpy()[h.item].tolist())
rows.append(sample(0));dt=.000125
for i in range(160):
 if mode=='full_graph':
  if i%8:continue
  h.advance(.001,dt=dt);rows.append(sample((i+8)*dt));continue
 if mode.startswith('full'):h.step(dt)
 else:
  h.state.clear_forces();h.control.joint_f.zero_();h.collision.collide(h.state,h.raw_contacts)
  h.rigid.step(h.state,h.next,h.control,h.raw_contacts,dt);h.state,h.next=h.next,h.state
 if i%4==3:rows.append(sample((i+1)*dt))
np.savez_compressed(OUT/f'penetration_{mode}_{frame}.npz',body_q=h.state.body_q.numpy(),joint_q=h.state.joint_q.numpy(),particles=h.state.particle_q.numpy()[:h.real_count])
res=dict(scope=__doc__,mode=mode,source_frame=frame,dt_s=dt,rows=rows,all_finite=bool(np.isfinite(np.array([r['stone_in_thumb'] for r in rows])).all()),object_motion_mm=((np.array(rows[-1]['stone_world'])-rows[0]['stone_world'])*1000).tolist(),relative_motion_mm=((np.array(rows[-1]['stone_in_thumb'])-rows[0]['stone_in_thumb'])*1000).tolist())
(OUT/f'penetration_{mode}_{frame}.json').write_text(json.dumps(res,indent=2));print('RELAX_RESULT',json.dumps({k:v for k,v in res.items() if k!='rows'}),flush=True)
