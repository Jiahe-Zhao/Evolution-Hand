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
from contact_guard_coordinated import install,install_controller
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
base_target=np.clip(cp['joint_q'][h.action_indices]*h.signs,h.action_lower,h.action_upper)
h.set_targets(base_target);wp.copy(h.guard_previous,h.target);wp.copy(h.guard_held,h.target)
h.guard_latched.assign((h.guard_finger.numpy()>=0).astype(np.int32));h.guard_enabled.assign(np.array([1],dtype=np.int32))
thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'));index=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
local_z=Rotation.from_quat(c['orientation']).inv().apply([0,0,1.])
period=.005;trigger=None;full_start=None;stable=0.;probe_anchor=None;probe_stone=None;max_native=np.zeros(2);rows=[];bodies=[];joints=[];particles=[];targets=[];contact_rows=[];stop='timeout';command_lift=0.
(OUT/f'{TAG}_initialization.json').write_text(json.dumps(dict(scope='continuation from measured 3.85s full state with velocities/feedback preserved; solver warm-start not restored; not a fresh full-task acceptance',source='jitter_source_state_20261127.npz',mode=mode),indent=2))

import mujoco as mj
from scipy.optimize import least_squares
amp=float(os.environ.get('INWARD_MM','0.25'))
m=mj.MjModel.from_binary_path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
for ai,name in enumerate(h.action_names):
 matches=[j for j in range(m.njnt) if m.joint(j).name.endswith('_Joints_'+name)]
 if matches:d.qpos[m.jnt_qposadr[matches[0]]]=cp['joint_q'][h.action_indices[ai]]*h.signs[ai]
js=np.array([next(j for j in range(m.njnt) if m.joint(j).name.endswith('_Joints_'+n)) for n in search['fingers']['index']['names']]);addr=m.jnt_qposadr[js];ais=np.array(search_indices['index']);seed=d.qpos[addr].copy();bid=next(j for j in range(m.nbody) if m.body(j).name.endswith('link_2_3'));local=np.array(c['local_points'][1]);mj.mj_kinematics(m,d)
b0=cp['body_q'];align=Rotation.from_quat(b0[index,3:])*Rotation.from_matrix(d.xmat[bid].reshape(3,3)).inv();origin=b0[index,:3]-align.apply(d.xpos[bid])
def fk(v):
 d.qpos[addr]=v;mj.mj_kinematics(m,d);return origin+align.apply(d.xpos[bid]+d.xmat[bid].reshape(3,3)@local)
start=fk(seed);thumb_point=b0[thumb,:3]+Rotation.from_quat(b0[thumb,3:]).apply(c['local_points'][0]);axis=thumb_point-start;axis/=np.linalg.norm(axis)
fit=least_squares(lambda v:np.r_[(fk(v)-start-axis*amp/1000)*1000,(v-seed)*.01],seed,bounds=(m.jnt_range[js,0]+1e-5,m.jnt_range[js,1]-1e-5),max_nfev=200)
ik_error=float(np.linalg.norm(fk(fit.x)-start-axis*amp/1000)*1000)
# Existing substep force/depth retreat remains. Feed small held-target increments,
# bounded by actual q +/- .003 and native joint limits on each physics step.
prev_fraction=0.;peak_actual=0.;lift_anchor=None;progress_mm=0.;filtered=0.;stable=0.;trigger=None
for i in range(240):
 t=i*period;phase='settle' if t<.05 else 'coordinated_close'
 if t>=.05 and trigger is None:
  if filtered<.05:progress_mm=min(amp,progress_mm+1.0*period)
  elif filtered>.2:progress_mm=max(0.,progress_mm-.2*period)
 fraction=progress_mm/amp
 target=base_target.copy();target[ais]=seed+(fit.x-seed)*fraction
 if trigger is not None:
  target[:3]+=local_z*.002*smooth((t-trigger)/.2);phase='micro_lift' if t-trigger<.2 else 'micro_hold'
 h.set_targets(target);h.guard_peak.zero_();h.guard_integral.zero_();h.guard_depth_peak.zero_();h.advance(period,dt=dt)
 means=h.guard_integral.numpy()/period;peaks=h.guard_peak.numpy();depth=h.guard_depth_peak.numpy()*1000;max_native=np.maximum(max_native,depth);filtered+=(1-np.exp(-period/.02))*(float(means[1])-filtered)
 b=h.state.body_q.numpy();q=h.state.joint_q.numpy();particles_now=h.state.particle_q.numpy();cs=summarize(i)
 tip=b[index,:3]+Rotation.from_quat(b[index,3:]).apply(local);delta=tip-start;actual=float(delta@axis*1000);peak_actual=max(peak_actual,actual)
 row=dict(time_s=h.time,phase=phase,actual_inward_mm=actual,off_axis_mm=float(np.linalg.norm(delta-axis*(delta@axis))*1000),thumb_mean_N=float(means[0]),index_mean_N=float(means[1]),soft_peak_N=float(peaks[2]),thumb_depth_mm=float(depth[0]),index_depth_mm=float(depth[1]),stone_x=float(b[h.item,0]),stone_y=float(b[h.item,1]),stone_z=float(b[h.item,2]),object_support_N=cs['object_support_N'],other_hand_item_N=cs['other_hand_item_N'],progress_command_mm=progress_mm,index_filtered_N=filtered)
 rows.append(row);bodies.append(b);joints.append(q);particles.append(particles_now[:h.real_count]);targets.append(h.target.numpy()[h.action_indices]*h.signs);contact_rows.append(cs)
 if i%20==0:print('STEP',row,flush=True)
 if not np.isfinite(b).all() or not np.isfinite(particles_now).all():stop='nonfinite';break
 if max(depth)>.8:stop='excessive_penetration';break
 if max(cs['neighbor1_displacement_mm'],cs['neighbor2_displacement_mm'])>5:stop='neighbor_displacement';break
 stable=stable+period if min(means[:2])>.005 and cs['other_hand_item_N']<1e-6 else 0.
 if trigger is None and t>.05 and stable>=.1:
  trigger=h.time;lift_anchor=(float(b[h.item,2]),q[:3].copy());print('TRIGGER',trigger,flush=True)
 if trigger is not None and h.time-trigger>=.4:stop='micro_probe_completed';break
 stop='dual_contact_not_sustained'
a=np.array([[x['stone_x'],x['stone_y'],x['stone_z']] for x in rows if x['time_s']>=.35]);recent=[x for x in rows if .35<=x['time_s']<=.5]
r=dict(requested_inward_mm=amp,ik_error_mm=ik_error,reference='body attached index fingertip point, not deformed soft surface',axis_world=axis.tolist(),actual_inward_peak_mm=peak_actual,actual_inward_final_mm=rows[-1]['actual_inward_mm'],off_axis_final_mm=rows[-1]['off_axis_mm'],stop=stop,hold_stone_span_mm=(np.ptp(a,axis=0)*1000).tolist(),hold_dual_fraction=float(np.mean([min(x['thumb_mean_N'],x['index_mean_N'])>.005 for x in recent])),max_native_depth_mm=max_native.tolist(),max_soft_reaction_N=max(x['soft_peak_N'] for x in rows),micro_lift_attempted=lift_anchor is not None,acceptance_passed=False,scope='fixed thumb reference and force-regulated index closure; same checkpoint, warmstart reset, not full acceptance')
r.update(trigger_s=trigger,final_progress_command_mm=progress_mm,final_index_filtered_N=filtered)
if lift_anchor:r.update(stone_rise_mm=(float(b[h.item,2])-lift_anchor[0])*1000,wrist_rise_mm=float(Rotation.from_quat(c['orientation']).apply(q[:3]-lift_anchor[1])[2]*1000))
with (OUT/f'{TAG}_trace.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(OUT/f'{TAG}.npz',time=[x['time_s'] for x in rows],phase=[x['phase'] for x in rows],body_q=bodies,joint_q=joints,particles=particles,targets=targets,body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contact_rows)
(OUT/f'{TAG}_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
