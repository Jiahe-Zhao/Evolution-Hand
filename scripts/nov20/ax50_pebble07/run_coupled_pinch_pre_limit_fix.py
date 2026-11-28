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
from contact_guard_geometry import install,install_controller
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

with (OUT/f'{TAG}_trace.csv').open('w') as f:
    fields = ['time_s','phase','finite','soft_max_speed_m_s','target_displacement_mm','lift_mm',
              'neighbor1_displacement_mm','neighbor2_displacement_mm','hand_joint_max_error_rad',
              'thumb_terminal_N','index_terminal_N','other_hand_item_N','neighbor_hand_contacts','dual_terminal_contact','object_support_N','hand_tray_N','thumb_search_mm','index_search_mm']
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i in range(round(max_duration/.01)):
        t=i*.01;wrist=base.copy();target=qtarget.copy()
        if t>=3.85 and lift_start is None and qualified_contact_s<.1:
            stop='dual_contact_not_established';break
        if t<1.2:
            phase='preshape_above';wrist[2]+=.05
            target[22:]=qtarget[22:]*smooth(t/.4)
            target[6:10]=qtarget[6:10]*smooth((t-.4)/.4)
            target[10:22]=qtarget[10:22]*smooth((t-.8)/.4)
        elif t<2.:
            phase='slow_descent';wrist[2]+=.05*(1-smooth((t-1.2)/.8))
        elif t<2.5:
            phase='close_pregrasp';closing=smooth((t-2.)/.5)
            if wrist_fraction is not None:closing=min(closing,wrist_fraction)
            target=qtarget+(qclose-qtarget)*closing;wrist+=close_delta*closing
        else:
            wrist+=close_delta*(wrist_fraction if wrist_fraction is not None else 1.);target=qclose.copy()
            if t<2.8:phase='pregrasp_settle'
            elif t<3.8:phase='position_clamp'
            elif t<3.85:phase='clamp_hold'
            elif t<4.65:phase='lift_30mm'
            else:phase='lift_hold'
            closing_fraction=smooth((t-2.8)/1.)
            for key in travel:
                amount=({'thumb':.65,'index':1.8}[key])*closing_fraction
                travel[key]=float(amount)
                for j,action in enumerate(search_indices[key]):target[action]=np.interp(amount,search_axes,search_values[key][:,j])
            if t>=3.85:
                lift_start=3.85
                wrist[2]+=.03*smooth((t-3.85)/.8)
        target[:3] = Rotation.from_quat(c['orientation']).inv().apply(wrist-base-np.array([0.,0.,.05]))
        h.guard_enabled.assign(np.array([int(t>=2. and force_enabled)],dtype=np.int32))
        h.set_targets(target)
        h.peak_rigid_load.zero_();h.peak_soft_load.zero_()
        h.force_components_window.zero_();h.guard_depth_peak.zero_();h.guard_peak.zero_();h.guard_integral.zero_()
        h.advance(.01, dt=dt)
        depth_rows.append(h.guard_depth_peak.numpy().tolist())
        peaks=h.guard_peak.numpy();means=h.guard_integral.numpy()/.01
        latched=h.guard_latched.numpy()
        if wrist_fraction is None and np.any(latched):wrist_fraction=float(smooth((t-2.)/.5))
        control_rows.append([h.time,*peaks,*means,int(np.any(latched[h.guard_finger.numpy()==0])),int(np.any(latched[h.guard_finger.numpy()==1]))])
        target=h.target.numpy()[h.action_indices]*h.signs
        rigid_peaks.append(h.peak_rigid_load.numpy());soft_peaks.append(h.peak_soft_load.numpy())
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
        qualified_contact_s=qualified_contact_s+.01 if min(means[:2])>.005 and cs['other_hand_item_N']<.005 else 0.

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
        if t<3.85 and disp[0]>20 and not cs['dual_terminal_contact']:
            stop='target_ejected_before_lift';break
        if not finite or speed>20:
            stop='nonfinite_or_runaway';break
        if np.max(disp[1:])>5:
            stop='neighbor_displacement_over_5mm';break

np.savetxt(OUT/f'{TAG}_controller.csv',np.asarray(control_rows),delimiter=',',header='time_s,thumb_peak_N,index_peak_N,soft_reaction_peak_N,soft_normal_sum_peak_N,thumb_mean_N,index_mean_N,soft_reaction_mean_N,soft_normal_sum_mean_N,thumb_latched,index_latched',comments='')
np.savez_compressed(OUT/f'{TAG}_final_state.npz',time=h.time,body_q=h.state.body_q.numpy(),body_qd=h.state.body_qd.numpy(),joint_q=h.state.joint_q.numpy(),joint_qd=h.state.joint_qd.numpy(),particle_q=h.state.particle_q.numpy(),particle_qd=h.state.particle_qd.numpy(),feedback=h.feedback.numpy(),target=h.target.numpy())

np.savez_compressed(OUT/f'{TAG}.npz',time=[r['time_s'] for r in rows],phase=[r['phase'] for r in rows],
                    body_q=bodies,joint_q=joints,particles=particles,targets=targets,
                    rigid_external_peak_N=rigid_peaks,soft_external_peak_N=soft_peaks,body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contact_rows,
                    thumb_terminal_N=[r['thumb_terminal_N'] for r in rows],index_terminal_N=[r['index_terminal_N'] for r in rows],
                    other_hand_item_N=[r['other_hand_item_N'] for r in rows])
result=dict(object_xml=str(object_xml),scope='actual short staged GPU trial; qualification does not imply task acceptance',stop=stop,time=h.time,
            max_object_displacement_mm=maximum.tolist(),max_soft_speed_m_s=max(r['soft_max_speed_m_s'] for r in rows),
            final_hand_joint_max_error_rad=rows[-1]['hand_joint_max_error_rad'],action_names=h.action_names,
            finite=all(r['finite'] for r in rows),dual_contact_samples=sum(r['dual_terminal_contact'] for r in rows),
            max_lift_mm=max(r['lift_mm'] for r in rows),max_neighbor_displacement_mm=float(max(maximum[1:])),
            joint_margins_rad=c['joint_margins_rad'],runtime_target=qtarget.tolist(),
            runtime_min_margin_rad=float(np.min(np.minimum(qtarget-h.action_lower,h.action_upper-qtarget))),
            measured_local_points=c['local_points'],contact_forces_available=True,search_travel_mm=travel,lift_start=lift_start,soft_kd_override=.5,components_at_peak_total=h.force_components_peak.numpy().tolist(),qualified_contact_s=qualified_contact_s,controller=dict(max_native_depth_mm=(np.max(depth_rows,axis=0)*1000).tolist(),geometry_guard=True,dt_s=dt,control_period_s=dt,force_guard=force_enabled,peak_target_forces_N=np.max(np.array(control_rows)[:,1:5],axis=0).tolist(),wrist_fraction=wrist_fraction),position_control=dict(active_joint_torque_cap_Nm=.02,active_joint_names=active_names,reference_closure_mm=dict(thumb=.65,index=1.8),clamp_duration_s=1.,clamp_hold_s=.05,wrist_lift_mm=30,lift_duration_s=.8,lift_hold_s=.6,force_gate=True,force_abort=False,rate_limit_rad_s=.25,contact_hold_threshold_N=.15))
(OUT/f'{TAG}_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
