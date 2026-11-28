"""Substep guarded closure; unchanged object/contact physics, fresh initial state."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
from pebble_runtime import *
import csv
import os
from contact_guard_geometry_bounded import install,install_controller
from contact_review_probe import ax50_contacts
from scipy.spatial.transform import Rotation

TAG = os.environ['RUN_TAG']
object_xml=Path(os.environ['OBJECT_XML']).resolve()
h = PebbleRuntime(side='left', wrist='six_axis', position=(np.asarray(json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['position'])+np.array([0.,0.,.05])),
                  orientation=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['orientation'],
                  gravity=(0, 0, -9.81), object_xml=object_xml)
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
install_probe(h,.5)
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


mode=os.environ.get('PROBE_MODE','lift')
cp=np.load(OUT/'jitter_source_state_20261127.npz')
for state in [h.state,h.next]:
 for key in ['body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd']:getattr(state,key).assign(cp[key])
h.feedback.assign(cp['feedback'])
base_target=np.clip(cp['target'][h.action_indices]*h.signs,h.action_lower,h.action_upper)
h.set_targets(base_target);wp.copy(h.guard_previous,h.target);wp.copy(h.guard_held,h.target)
h.guard_latched.assign((h.guard_finger.numpy()>=0).astype(np.int32));h.guard_enabled.assign(np.array([1],dtype=np.int32))
thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'));index=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
local_z=Rotation.from_quat(c['orientation']).inv().apply([0,0,1.])
period=.005;trigger=None;full_start=None;stable=0.;probe_anchor=None;probe_stone=None;max_native=np.zeros(2);rows=[];bodies=[];joints=[];particles=[];targets=[];contact_rows=[];stop='timeout';command_lift=0.
(OUT/f'{TAG}_initialization.json').write_text(json.dumps(dict(scope='continuation from measured 3.85s full state with velocities/feedback preserved; solver warm-start not restored; not a fresh full-task acceptance',source='jitter_source_state_20261127.npz',mode=mode),indent=2))
for i in range(500):
 t=i*period;command_lift=0.;phase='contact_settle' if t<.05 else 'establish_contact'
 if trigger is not None:
  u=t-trigger
  if mode=='hold':phase='matched_hold'
  else:
   command_lift=.002*smooth(u/.2);phase='micro_lift' if u<.2 else 'micro_hold'
   if full_start is not None:command_lift=.002+.020*smooth((t-full_start)/.8);phase='continue_lift' if t-full_start<.8 else 'lift_hold'
 target=base_target.copy();target[:3]+=local_z*command_lift;h.set_targets(target)
 h.guard_peak.zero_();h.guard_integral.zero_();h.guard_depth_peak.zero_();h.advance(period,dt=.000125)
 means=h.guard_integral.numpy()/period;peaks=h.guard_peak.numpy();depth=h.guard_depth_peak.numpy()*1000;max_native=np.maximum(max_native,depth)
 b=h.state.body_q.numpy();q=h.state.joint_q.numpy();p=h.state.particle_q.numpy();cs=summarize(i)
 center=(b[thumb,:3]+b[index,:3])/2;rel=Rotation.from_quat(b[thumb,3:]).inv().apply(b[h.item,:3]-center)
 finite=all(np.isfinite(x).all() for x in [b,q,p,h.state.body_qd.numpy(),h.state.particle_qd.numpy()])
 neighbors=float(np.max(np.linalg.norm(b[h.object_bodies[1:],:3]-initial[1:],axis=1))*1000)
 row=dict(time_s=h.time,phase=phase,finite=finite,lift_mm=cs['lift_mm'],neighbor1_displacement_mm=cs['neighbor1_displacement_mm'],neighbor2_displacement_mm=cs['neighbor2_displacement_mm'],thumb_terminal_N=cs['thumb_terminal_N'],index_terminal_N=cs['index_terminal_N'],other_hand_item_N=cs['other_hand_item_N'],object_support_N=cs['object_support_N'],hand_tray_N=cs['hand_tray_N'],thumb_mean_N=float(means[0]),index_mean_N=float(means[1]),soft_peak_N=float(peaks[2]),thumb_depth_mm=float(depth[0]),command_lift_mm=command_lift*1000,stone_x=float(b[h.item,0]),stone_y=float(b[h.item,1]),stone_z=float(b[h.item,2]),center_z=float(center[2]),relative_x=float(rel[0]),relative_y=float(rel[1]),relative_z=float(rel[2]))
 rows.append(row);bodies.append(b);joints.append(q);particles.append(p[:h.real_count]);targets.append(h.target.numpy()[h.action_indices]*h.signs);contact_rows.append(cs)
 if t>=.05 and trigger is None:
  stable=stable+period if min(means[:2])>=.005 and cs['other_hand_item_N']<=1e-6 else 0.
  if stable>=.02-1e-9:trigger=h.time;probe_anchor=float(center[2]);probe_stone=float(b[h.item,2]);print('TRIGGER',trigger,means[:2].tolist(),flush=True)
 if i%20==0:print('STEP',row,flush=True)
 if not finite:stop='nonfinite';break
 if neighbors>5:stop='neighbor_displacement';break
 if trigger is None and h.time>=.6:stop='no_sustained_dual_contact';break
 if trigger is not None and mode=='hold' and h.time-trigger>=.45:stop='matched_hold_completed';break
 if trigger is not None and mode!='hold' and full_start is None and h.time-trigger>=.4:
  recent=rows[-10:];off_tray=all(r['object_support_N']<=1e-5 for r in recent)
  hand_rise=float(center[2])-probe_anchor;stone_rise=float(b[h.item,2])-probe_stone
  follows=hand_rise>=.0015 and stone_rise>=.001 and abs(stone_rise-hand_rise)<=.00075
  contact=min(np.mean([[r['thumb_mean_N'],r['index_mean_N']] for r in recent],axis=0))>=.005
  if off_tray and follows and contact:full_start=h.time;print('MICRO_LIFT_PASSED',hand_rise,stone_rise,flush=True)
  else:stop='micro_lift_not_stable';break
 if full_start is not None and h.time-full_start>=1.15:stop='lift_hold_completed';break
np.savez_compressed(OUT/f'{TAG}.npz',time=[r['time_s'] for r in rows],phase=[r['phase'] for r in rows],body_q=bodies,joint_q=joints,particles=particles,targets=targets,body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contact_rows,thumb_terminal_N=[r['thumb_terminal_N'] for r in rows],index_terminal_N=[r['index_terminal_N'] for r in rows],other_hand_item_N=[r['other_hand_item_N'] for r in rows])
with (OUT/f'{TAG}_trace.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r=dict(scope='paired continuation diagnostic; not fresh full-task acceptance',mode=mode,object_xml=str(object_xml),stop=stop,time=h.time,trigger_s=trigger,full_lift_start_s=full_start,max_lift_mm=max(x['lift_mm'] for x in rows),max_neighbor_displacement_mm=max(max(x['neighbor1_displacement_mm'],x['neighbor2_displacement_mm']) for x in rows),max_native_depth_mm=max_native.tolist(),max_soft_reaction_N=max(x['soft_peak_N'] for x in rows),finite=all(x['finite'] for x in rows),acceptance_passed=False)
if trigger is not None:r.update(hand_rise_mm=(float(center[2])-probe_anchor)*1000,stone_rise_mm=(float(b[h.item,2])-probe_stone)*1000)
(OUT/f'{TAG}_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
