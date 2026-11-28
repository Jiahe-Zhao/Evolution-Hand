"""Recorded staged pebble pinch: preshape above, descend, contact, lift only when qualified."""

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
from contact_review_probe import ax50_contacts
from scipy.spatial.transform import Rotation

TAG = os.environ.get('RUN_TAG', 'pinch_trial_v5')
h = PebbleRuntime(side='left', wrist='six_axis', position=(np.asarray(json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['position'])+np.array([0.,0.,.05])),
                  orientation=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['orientation'],
                  gravity=(0, 0, -9.81))
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
checkpoint_saved=False



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

with (OUT/f'{TAG}_trace.csv').open('w') as f:
    fields = ['time_s','phase','finite','soft_max_speed_m_s','target_displacement_mm','lift_mm',
              'neighbor1_displacement_mm','neighbor2_displacement_mm','hand_joint_max_error_rad',
              'thumb_terminal_N','index_terminal_N','other_hand_item_N','neighbor_hand_contacts','dual_terminal_contact','object_support_N','hand_tray_N','thumb_search_mm','index_search_mm']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i in range(901):
        t=i*.01;wrist=base.copy();target=qtarget.copy()
        if t<1.2:
            phase='preshape_above';wrist[2]+=.05
            target[22:]=qtarget[22:]*smooth(t/.4)
            target[6:10]=qtarget[6:10]*smooth((t-.4)/.4)
            target[10:22]=qtarget[10:22]*smooth((t-.8)/.4)
        elif t<2.:
            phase='slow_descent';wrist[2]+=.05*(1-smooth((t-1.2)/.8))
        elif t<2.5:
            phase='close_pregrasp';closing=smooth((t-2.)/.5)
            target=qtarget+(qclose-qtarget)*closing;wrist+=close_delta*closing
        else:
            phase='force_limited_search';target=qclose.copy();wrist+=close_delta
            # 100 Hz contact feedback, 0.8 mm/s search, 0.4 mm/s release.
            # Thresholds control observed load, not an instantaneous force guarantee.
            for key in travel:
                rate=.8 if force[key]<.04 else (-.4 if force[key]>.10 else 0.)
                travel[key]=float(np.clip(travel[key]+rate*.01,0,search_axes[-1]))
                for j,action in enumerate(search_indices[key]):
                    target[action]=np.interp(travel[key],search_axes,search_values[key][:,j])
            if lift_start is not None:
                elapsed=t-lift_start
                phase='lift_20mm' if elapsed<.8 else 'lift_hold'
                wrist[2]+=.03*smooth(elapsed/.8)
        target[:3] = Rotation.from_quat(c['orientation']).inv().apply(wrist-base-np.array([0.,0.,.05]))
        h.set_targets(target)
        h.advance(.01, dt=.000125)
        b = h.state.body_q.numpy()
        q = h.state.joint_q.numpy()
        p = h.state.particle_q.numpy()
        vel = h.state.particle_qd.numpy()
        finite = all(np.isfinite(x).all() for x in [b,q,p,vel,h.state.joint_qd.numpy()])
        speed = float(np.max(np.linalg.norm(vel,axis=1)))
        disp=np.linalg.norm(b[h.object_bodies,:3]-initial,axis=1)*1000
        maximum=np.maximum(maximum,disp)
        error=float(np.max(np.abs(q[h.action_indices[6:]]-target[6:])))
        cs = summarize(i)
        force=dict(thumb=cs['thumb_terminal_N'],index=cs['index_terminal_N'])
        tray_frames=tray_frames+1 if cs['hand_tray_N']>1. else 0
        row = dict(time_s=h.time,phase=phase,finite=finite,soft_max_speed_m_s=speed,
                   target_displacement_mm=float(disp[0]),lift_mm=cs['lift_mm'],neighbor1_displacement_mm=float(disp[1]),
                   neighbor2_displacement_mm=float(disp[2]),hand_joint_max_error_rad=error,
                   thumb_terminal_N=cs['thumb_terminal_N'],index_terminal_N=cs['index_terminal_N'],
                   other_hand_item_N=cs['other_hand_item_N'],neighbor_hand_contacts=cs['neighbor_hand_contacts'],
                   dual_terminal_contact=cs['dual_terminal_contact'],object_support_N=cs['object_support_N'],hand_tray_N=cs['hand_tray_N'],thumb_search_mm=travel['thumb'],index_search_mm=travel['index'])
        rows.append(row);w.writerow({k:row[k] for k in fields});f.flush()
        bodies.append(b);joints.append(q);particles.append(p[:h.real_count]);targets.append(target);contact_rows.append(cs)
        if i%10==0: print('STEP',row,flush=True)
        if tray_frames>=5:
            stop='hand_tray_contact_over_1N_for_50ms';break
        if not finite or speed>20:
            stop='nonfinite_or_runaway';break
        if np.max(disp[1:])>5:
            stop='neighbor_displacement_over_5mm';break
        if max(force.values())>.4:
            stop='sampled_terminal_force_over_0.4N';break
        if t>=2.5:
            dual=all(v>=.025 for v in force.values()) and cs['other_hand_item_N']<1e-6 and cs['neighbor_hand_contacts']==0
            contact_since=(h.time if contact_since is None else contact_since) if dual else None
            if lift_start is None and contact_since is not None and h.time-contact_since>=.15:
                lift_start=h.time;print('LIFT_GATE',h.time,force,travel,flush=True)
            if lift_start is not None and t-lift_start>=1.3:
                stop='lift_and_hold_completed';break
            if lift_start is None and t>=7.6:
                stop='search_timeout_no_sustained_dual_contact';break
        if t>=2.5 and not checkpoint_saved:
            np.savez_compressed(OUT/f'{TAG}_pregrasp_state.npz',time=h.time,body_q=b,body_qd=h.state.body_qd.numpy(),joint_q=q,joint_qd=h.state.joint_qd.numpy(),particle_q=h.state.particle_q.numpy(),particle_qd=h.state.particle_qd.numpy(),targets=target)
            checkpoint_saved=True

np.savez_compressed(OUT/f'{TAG}.npz',time=[r['time_s'] for r in rows],phase=[r['phase'] for r in rows],
                    body_q=bodies,joint_q=joints,particles=particles,targets=targets,
                    body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contact_rows,
                    thumb_terminal_N=[r['thumb_terminal_N'] for r in rows],index_terminal_N=[r['index_terminal_N'] for r in rows],
                    other_hand_item_N=[r['other_hand_item_N'] for r in rows])
result=dict(scope='actual short staged GPU trial; qualification does not imply task acceptance',stop=stop,time=h.time,
            max_object_displacement_mm=maximum.tolist(),max_soft_speed_m_s=max(r['soft_max_speed_m_s'] for r in rows),
            final_hand_joint_max_error_rad=rows[-1]['hand_joint_max_error_rad'],action_names=h.action_names,
            finite=all(r['finite'] for r in rows),dual_contact_samples=sum(r['dual_terminal_contact'] for r in rows),
            max_lift_mm=max(r['lift_mm'] for r in rows),max_neighbor_displacement_mm=float(max(maximum[1:])),
            joint_margins_rad=c['joint_margins_rad'],runtime_target=qtarget.tolist(),
            runtime_min_margin_rad=float(np.min(np.minimum(qtarget-h.action_lower,h.action_upper-qtarget))),
            measured_local_points=c['local_points'],contact_forces_available=True,search_travel_mm=travel,lift_start=lift_start,force_control=dict(sample_hz=100,search_mm_s=.8,release_mm_s=.4,band_N=[.04,.10],abort_sample_N=.4))
(OUT/f'{TAG}_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
