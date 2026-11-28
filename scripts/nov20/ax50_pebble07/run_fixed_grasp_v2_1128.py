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
h = PebbleRuntime(side='left', wrist='six_axis', position=(np.asarray(json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())['position'])+np.array([0.,0.,.05])),
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
install_probe(h,.5)
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
from scipy.optimize import least_squares
import mujoco as mj
from soft_depth_guard import install as install_soft_guard
install_soft_guard(h)
thumb=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_1_2'))
index=next(i for i,n in enumerate(h.model.body_label) if str(n).endswith('/link_2_3'))
rest_pose=h.model.body_q.numpy()[index];rest_local=R.from_quat(rest_pose[3:]).inv().apply(h.model.particle_q.numpy()[:h.real_count]-rest_pose[:3])
cp=np.load(OUT/'jitter_source_state_20261127.npz')
for state in [h.state,h.next]:
 for key in ['body_q','body_qd','joint_q','joint_qd','particle_q','particle_qd']:getattr(state,key).assign(cp[key])
h.feedback.zero_()
initial=h.state.body_q.numpy()[h.object_bodies,:3].copy()
base_target=np.clip(cp['target'][h.action_indices]*h.signs,h.action_lower,h.action_upper)
for name in active_names:
 ai=h.action_names.index(name);base_target[ai]=np.clip(cp['joint_q'][h.action_indices[ai]]*h.signs[ai],h.action_lower[ai],h.action_upper[ai])
h.set_targets(base_target);wp.copy(h.guard_previous,h.target);wp.copy(h.guard_held,h.target);h.guard_enabled.assign(np.array([1],dtype=np.int32))
m=mj.MjModel.from_binary_path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
for ai,name in enumerate(h.action_names):
 js=[j for j in range(m.njnt) if m.joint(j).name.endswith('_Joints_'+name)]
 if js:d.qpos[m.jnt_qposadr[js[0]]]=base_target[ai]
seeds=d.qpos.copy();mj.mj_kinematics(m,d);body0=h.state.body_q.numpy();fits={};points=[]
for fi,(key,body) in enumerate([('thumb',thumb),('index',index)]):points.append(body0[body,:3]+R.from_quat(body0[body,3:]).apply(c['local_points'][fi]))
axis=points[0]-points[1];axis[2]=0;axis/=np.linalg.norm(axis)
for fi,(key,body) in enumerate([('thumb',thumb),('index',index)]):
 d.qpos[:]=seeds;mj.mj_kinematics(m,d);js=np.array([next(j for j in range(m.njnt) if m.joint(j).name.endswith('_Joints_'+n)) for n in search['fingers'][key]['names']]);addr=m.jnt_qposadr[js];ais=np.array(search_indices[key]);seed=d.qpos[addr].copy();bid=next(j for j in range(m.nbody) if m.body(j).name.endswith('link_1_2' if fi==0 else 'link_2_3'));local=np.array(c['local_points'][fi]);align=R.from_quat(body0[body,3:])*R.from_matrix(d.xmat[bid].reshape(3,3)).inv();origin=body0[body,:3]-align.apply(d.xpos[bid]);direction=axis*(-1 if fi==0 else 1)
 def fk(v):
  d.qpos[addr]=v;mj.mj_kinematics(m,d);return origin+align.apply(d.xpos[bid]+d.xmat[bid].reshape(3,3)@local)
 start=fk(seed);goal=start+direction*.002
 fit=least_squares(lambda v:np.r_[(fk(v)-goal)*1000,(v-seed)*.01],seed,bounds=(m.jnt_range[js,0]+1e-5,m.jnt_range[js,1]-1e-5),max_nfev=300)
 err=float(np.linalg.norm(fk(fit.x)-goal)*1000);assert err<.1,err
 fits[key]=(ais,seed,fit.x,err)
local_z=R.from_quat(c['orientation']).inv().apply([0,0,1.]);period=.005;travel=np.zeros(2);filtered=np.zeros(2);trigger=None;lift_ref=None;stable=0.;held=0.;rows=[];bodies=[];particles=[];joints=[];targets=[];contact_rows=[];friction_samples=[];stop='grip_not_stable';duration=float(os.environ.get('MAX_DURATION','4.5'))
(OUT/(TAG+'_initialization.json')).write_text(json.dumps(dict(source='jitter_source_state_20261127.npz',scope='checkpoint continuation; feedback cleared; warmstart not restored; 0.2 s settling; not fresh task acceptance',model=h.stable_contact_info,targets_N=[.65,.85],max_closure_each_mm=2,stable_gate='both >=0.15 N, CV <=0.2, stone span <=0.2 mm and speed <=3 mm/s, continuous 0.2 s',lift_mm=2,command_lift_mm=2.2,potential_vertical_support_gate_N=.65,hold_s=.5),indent=2))
for i in range(round(duration/period)):
 t=i*period;phase='settle' if t<.2 else 'close'
 if t>=.2:
  for k,goal in enumerate([.65,.85]):
   if filtered[k]<goal:travel[k]=min(2.,travel[k]+.8*period)
   elif filtered[k]>goal+.15:travel[k]=max(0.,travel[k]-.2*period)
 target=base_target.copy()
 for k,key in enumerate(['thumb','index']):
  ais,seed,end,err=fits[key];target[ais]=seed+(end-seed)*(travel[k]/2)
 if trigger is not None:
  target[:3]+=local_z*.0022*smooth((t-trigger)/.4);phase='lift' if t-trigger<.4 else 'hold'
 h.set_targets(target);h.guard_peak.zero_();h.guard_integral.zero_();h.guard_depth_peak.zero_();h.soft_depth_peak.zero_();h.vector_integral.zero_();h.advance(period,dt=dt)
 means=h.guard_integral.numpy()/period;filtered+=(1-np.exp(-period/.03))*(means[:2]-filtered);dep=h.guard_depth_peak.numpy()*1000;softdep=float(h.soft_depth_peak.numpy()[0]*1000);b=h.state.body_q.numpy();q=h.state.joint_q.numpy();ps=h.state.particle_q.numpy();cs=summarize(i);v=h.vector_integral.numpy()/period
 # Rest-relative, contact-normal deformation; separate from collision overlap.
 rest_world=b[index,:3]+R.from_quat(b[index,3:]).apply(rest_local);s=h.contacts;n=int(s.soft_contact_count.numpy()[0]);ii=s.soft_contact_indices.numpy()[:n];bb=s.soft_contact_barycentric.numpy()[:n];ss=s.soft_contact_shape.numpy()[:n];nn=s.soft_contact_normal.numpy()[:n];sb=h.model.shape_body.numpy();ff=h.out_f.numpy();deform=[];weights=[]
 for j in np.flatnonzero(sb[ss]==h.item):
  valid=ii[j]>=0;ids=ii[j,valid];w=bb[j,valid];pid=ids[np.argmax(w)]
  if pid>=h.real_count or h.regions[pid]!=3:continue
  fn=max(0.,float(ff[j]@nn[j]))
  if fn<=1e-7:continue
  delta=((ps[ids]-rest_world[ids])*w[:,None]).sum(0);deform.append(max(0.,float(delta@nn[j])*1000));weights.append(fn)
 rise=(b[h.item,2]-(lift_ref[0] if lift_ref is not None else initial[0,2]))*1000
 row=dict(time_s=h.time,phase=phase,thumb_mean_N=float(means[0]),index_mean_N=float(means[1]),thumb_depth_mm=float(dep[0]),index_depth_mm=float(dep[1]),soft_overlap_mm=softdep,soft_compression_max_mm=max(deform,default=0.),soft_compression_mean_mm=float(np.average(deform,weights=weights)) if weights else 0.,soft_peak_N=float(h.guard_peak.numpy()[2]),stone_x=float(b[h.item,0]),stone_y=float(b[h.item,1]),stone_z=float(b[h.item,2]),lift_mm=float(rise),stone_speed_m_s=float(np.linalg.norm(h.state.body_qd.numpy()[h.item,:3])),thumb_command_mm=float(travel[0]),index_command_mm=float(travel[1]),other_hand_item_N=cs['other_hand_item_N'],object_support_N=cs['object_support_N'],neighbor_displacement_mm=max(cs['neighbor1_displacement_mm'],cs['neighbor2_displacement_mm']))
 for gi,gn in enumerate(['thumb','index_rigid','index_soft','other','support']):
  for ki,kn in enumerate(['fx','fy','fz','nx','ny','nz','tx','ty','tz','active_count']):row[gn+'_'+kn]=float(v[gi,ki])
 row['wrist_rise_mm']=float(R.from_quat(c['orientation']).apply(q[:3]-lift_ref[1])[2]*1000) if lift_ref is not None else 0.
 rows.append(row);bodies.append(b);particles.append(ps[:h.real_count]);joints.append(q);targets.append(h.target.numpy()[h.action_indices]*h.signs);contact_rows.append(cs);friction_samples.append(snapshot(h))
 ssnap=np.asarray(friction_samples[-1],dtype=float).reshape(-1,18)
 soft_capacity_z=float(np.sum(ssnap[:,7]*np.sqrt(np.maximum(0.,1-ssnap[:,14]**2)))) if len(ssnap) else 0.
 row['potential_up_N']=float(v[0,5]+v[1,5]+v[2,5]+.9*np.linalg.norm(v[0,3:5])+.9*np.linalg.norm(v[1,3:5])+soft_capacity_z)
 if i%40==0:print('STEP',row,flush=True)
 if not np.isfinite(b).all() or not np.isfinite(ps).all():stop='nonfinite';break
 if max(dep)>.8 or softdep>.8:stop='penetration_stop';break
 if row['neighbor_displacement_mm']>5:stop='neighbor_stop';break
 window=rows[-40:];forces=np.array([[x['thumb_mean_N'],x['index_mean_N']] for x in window]);xyz=np.array([[x['stone_x'],x['stone_y'],x['stone_z']] for x in window]);good=len(window)==40 and np.min(forces)>.15 and np.max(forces.std(0)/forces.mean(0))<.2 and np.max(np.ptp(xyz,axis=0))*1000<.2 and row['stone_speed_m_s']<.003 and row['other_hand_item_N']<1e-6
 if trigger is None and t>.3 and good and min(x['potential_up_N'] for x in window)>.65:
  trigger=h.time;lift_ref=(float(b[h.item,2]),q[:3].copy());print('STABLE_GRIP_LIFT',trigger,flush=True)
 if trigger is not None:
  ok=rise>=2. and row['support_fz']<.01 and min(means[:2])>.005 and row['other_hand_item_N']<1e-6
  held=held+period if ok else 0.
  if held>=.5:stop='passed_2mm_hold';break
  if h.time-trigger>=1.1:stop='lift_not_held';break
r=dict(stop=stop,acceptance_passed=stop=='passed_2mm_hold',micro_lift_attempted=trigger is not None,trigger_s=trigger,stable_lift_hold_s=held,actual_mass_g=actual_mass*1000,object_xml=str(object_xml),contact_model=h.stable_contact_info,scope='50 g checkpoint continuation; 2 mm diagnostic, not original 20 mm full task',max_lift_mm=max(x['lift_mm'] for x in rows),final_lift_mm=rows[-1]['lift_mm'],max_soft_overlap_mm=max(x['soft_overlap_mm'] for x in rows),max_soft_compression_mm=max(x['soft_compression_max_mm'] for x in rows),max_rigid_depth_mm=max(max(x['thumb_depth_mm'],x['index_depth_mm']) for x in rows),max_neighbor_displacement_mm=max(x['neighbor_displacement_mm'] for x in rows),finite=bool(np.isfinite(bodies).all() and np.isfinite(particles).all()),ik_errors_mm={k:v[3] for k,v in fits.items()})
with (OUT/(TAG+'_trace.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(OUT/(TAG+'.npz'),time=[x['time_s'] for x in rows],phase=[x['phase'] for x in rows],body_q=bodies,joint_q=joints,particles=particles,targets=targets,body_names=h.model.body_label,tri=h.model.tri_indices.numpy(),contacts=contact_rows)
np.savez_compressed(OUT/(TAG+'_friction.npz'),time=[x['time_s'] for x in rows],soft=np.array(friction_samples,dtype=object))
(OUT/(TAG+'_result.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
