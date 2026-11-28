"""Substep guarded closure; unchanged object/contact physics, fresh initial state."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
from friction_audit import snapshot
friction_samples=[];native_samples=[]
from pebble_runtime import *
import csv
import os
from contact_guard_coordinated import install,install_controller
from contact_review_probe import ax50_contacts
from scipy.spatial.transform import Rotation

TAG = os.environ['RUN_TAG']
object_xml=Path(os.environ['OBJECT_XML']).resolve()
class PadRuntime(PebbleRuntime):
 def _add_brush(self,b):
  super()._add_brush(b)
  b.body_flags[self.item]=int(newton.BodyFlags.KINEMATIC)
h = PadRuntime(side='left', wrist='six_axis', position=(np.asarray(json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['position'])+np.array([0.,0.,.05])),
                  orientation=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['orientation'],
                  gravity=(0, 0, -9.81), object_xml=object_xml)
actual_mass=float(h.model.body_mass.numpy()[h.item]);expected_mass=float(os.environ['MASS_G'])/1000
assert abs(actual_mass-expected_mass)<1e-7,(actual_mass,expected_mass)
(OUT/f'{TAG}_mass_audit.json').write_text(json.dumps(dict(actual_mass_kg=actual_mass,inertia=h.model.body_inertia.numpy()[h.item].tolist(),object_xml=str(object_xml)),indent=2))
install(h)
c = json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())
initial = h.state.body_q.numpy()[h.object_bodies, :3].copy()
qtarget = np.zeros(len(h.action_names))
assert len(c['joint_names']) == len(c['joint_values'])
for name, value in zip(c['joint_names'], c['joint_values']):
    qtarget[h.action_names.index(name)] = value
qtarget = np.clip(qtarget, h.action_lower+.05, h.action_upper-.05)
close=json.loads((Path(__file__).parent/'candidates/pinch_close_candidate_v4.json').read_text())
assert close['accepted_for_trial']
qclose=qtarget.copy()
for name,value in zip(close['joint_names'],close['joint_values']):qclose[h.action_names.index(name)]=value
close_delta=np.array(close['wrist_delta_world'])
base = np.asarray(c['position'], float)
rows, bodies, joints, particles, targets, contact_rows = [], [], [], [], [], []
stop = 'completed'
maximum = np.zeros(3)
qualified_since = None
lift_start = None
smooth = lambda x: np.clip(x, 0, 1)**2 * (3 - 2*np.clip(x, 0, 1))
last_contacts = []
closing_latched=None
tray_frames=0
search=json.loads((Path(__file__).parent/'candidates/contact_search_v5.json').read_text())
travel={k:0. for k in ['thumb','index']}
force={k:0. for k in travel}
search_axes=np.array(search['travel_mm'])
search_indices={k:[h.action_names.index(n) for n in v['names']] for k,v in search['fingers'].items()}
search_values={k:np.array(v['joint_values']) for k,v in search['fingers'].items()}
contact_since=None
lift_start=None
checkpoint_saved=True
caps=h.caps.numpy()
active_names=search['fingers']['thumb']['names']+search['fingers']['index']['names']
for name in active_names:caps[h.action_indices[h.action_names.index(name)]]=.02
h.caps.assign(caps)
rigid_peaks=[];soft_peaks=[]
install_controller(h,search)
from soft_coupling_probe import install as install_probe
install_probe(h,float(os.environ.get("CONTACT_KD","0.5")))
from vector_probe import install as install_vector
install_vector(h)
qualified_contact_s=0.
dt=float(os.environ.get('SIM_DT','0.000125'))
max_duration=float(os.environ.get('MAX_DURATION','5.26'))
wrist_fraction=None
control_rows=[]
depth_rows=[]
force_enabled=os.environ.get('GUARD_ENABLED','1')=='1'

(OUT/f'{TAG}_initialization.json').write_text(json.dumps(dict(scope='fresh runtime; above-hand sequential preshape and descent; no checkpoint restoration',object_xml=str(object_xml)),indent=2))


def summarize(i):
    global last_contacts
    obs=h.rigid_observation
    count=int(obs.rigid_contact_count.numpy()[0]);sh0=obs.rigid_contact_shape0.numpy()[:count];sh1=obs.rigid_contact_shape1.numpy()[:count]
    forces=obs.force.numpy()[:count,:3];normals=obs.rigid_contact_normal.numpy()[:count]
    contacts=[dict(geom0=str(h.model.shape_label[int(a)]),geom1=str(h.model.shape_label[int(b)]),force_N=abs(float(f@n))) for a,b,f,n in zip(sh0,sh1,forces,normals)]
    sb=h.model.shape_body.numpy()
    for row,a,b in zip(contacts,sh0,sh1):row['hand_contact']=bool(0<=sb[int(a)]<h.hand_body_count or 0<=sb[int(b)]<h.hand_body_count)
    last_contacts=contacts
    bq = h.state.body_q.numpy()
    item = bq[h.item, :3] - initial[0]
    out = {'time_s': h.time, 'phase': phase, 'target_displacement_mm': float(np.linalg.norm(item)*1000),
           'lift_mm': float(item[2]*1000), 'neighbor1_displacement_mm': float(np.linalg.norm(bq[h.object_bodies[1],:3]-initial[1])*1000),
           'neighbor2_displacement_mm': float(np.linalg.norm(bq[h.object_bodies[2],:3]-initial[2])*1000),
           'thumb_terminal_N': 0., 'index_terminal_N': 0., 'other_hand_item_N': 0., 'neighbor_hand_contacts': 0,
           'dual_terminal_contact': False, 'contact_geoms': [], 'object_support_N':0., 'hand_tray_N':0.}
    for k, row in enumerate(contacts):
        force=row['force_N']
        if force <= 1e-6:continue
        pair = row['geom0'] + '|' + row['geom1']
        if row['hand_contact'] and any(n in pair for n in ['tray','rim','floor']):out['hand_tray_N']+=force
        if 'stone0' not in pair and 'foot0' not in pair:
            if 'stone1' in pair or 'stone2' in pair or 'foot1' in pair or 'foot2' in pair:
                if 'link_' in pair:
                    out['neighbor_hand_contacts'] += 1
            continue
        out['contact_geoms_force_available'] = True
        out['contact_geoms'].append({'pair': pair, 'force_N': force})
        if 'link_1_2' in pair:
            out['thumb_terminal_N'] += force
        elif 'link_2_3' in pair:
            out['index_terminal_N'] += force
        elif row['hand_contact']:
            out['other_hand_item_N'] += force
        else:out['object_support_N']+=force
    soft=h.contacts;n=int(soft.soft_contact_count.numpy()[0]);sh=soft.soft_contact_shape.numpy()[:n];ff=h.out_f.numpy()[:n];nn=soft.soft_contact_normal.numpy()[:n];ii=soft.soft_contact_indices.numpy()[:n];bb=soft.soft_contact_barycentric.numpy()[:n];sb=h.model.shape_body.numpy()
    for j in range(n):
        if int(sb[int(sh[j])]) != h.item:continue
        fn=abs(float(ff[j]@nn[j]))
        if fn<=1e-6:continue
        pid=int(ii[j,np.argmax(bb[j])]);region=h.region_names[h.regions[pid]] if 0<=pid<h.real_count and h.regions[pid]>=0 else 'unknown_soft'
        out['contact_geoms'].append(dict(pair='soft/'+region+'|'+str(h.model.shape_label[int(sh[j])]),force_N=fn))
        if region=='index_distal':out['index_terminal_N']+=fn
        else:out['other_hand_item_N']+=fn
    out['dual_terminal_contact'] = out['thumb_terminal_N'] > 0 and out['index_terminal_N'] > 0
    return out



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
from coupling_trace import install as trace_install
trace_install(h,axis)
from stable_soft_contact import install as stable_install
stable_install(h)
for it in range(340):
 t=it*period;slide=0.
 if t<.1:press=0.;phase='settle'
 elif t<.6:press=.0004*smooth((t-.1)/.5);phase='press'
 elif t<.8:press=.0004;phase='hold'
 elif t<1.2:press=.0004;slide=.002*smooth((t-.8)/.4);phase='slide'
 elif t<1.3:press=.0004;slide=.002;phase='post_slide'
 else:press=.0004*(1-smooth((t-1.3)/.4));slide=.002;phase='unload'
 target=target0.copy();target[:3]+=rot.inv().apply(axis*press+tangent*slide);h.set_targets(target);h.guard_peak.zero_();h.guard_depth_peak.zero_();h.vector_integral.zero_();h.advance(period,dt=dt)
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

np.savez_compressed(OUT/(TAG+"_substeps.npz"),data=h.subtrace.numpy()[:int(h.subcursor.numpy()[0])],dt=dt,columns=["normal","elastic","damping","active","depth_mm","mu_sum","kd_sum","anchor_axis","feedback_axis","bone_velocity_axis"])

(OUT/(TAG+"_fix.json")).write_text(json.dumps(dict(**h.stable_contact_info,duplicate_records_removed=int(h.dedup_removed.numpy()[0])),indent=2))
