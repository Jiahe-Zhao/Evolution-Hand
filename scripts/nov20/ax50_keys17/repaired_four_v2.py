
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from repair_keys_runtime import *
from key_monitor import AuditedKeys
import json,csv,os
out=ROOT/'outputs/nov20/ax50_keys17';plan=json.loads((out/'left_plan.json').read_text())
h=AuditedKeys(side='left',wrist='six_axis',position=(np.array(plan['position'])+[0,0,.04]).tolist(),orientation=plan['orientation'],gravity=(0,0,-9.81))
tag=os.environ.get('RUN_TAG','repaired_four')
if os.environ.get('VERTEX','0')=='1':
 h.collision=newton.CollisionPipeline(h.model,broad_phase='sap',rigid_contact_max=8192,soft_contact_max=131072,soft_contact_margin=.0005,enable_rigid_soft_full_surface_contact=False)
 h.raw_contacts=h.collision.contacts();h.contacts=h.collision.contacts()
 for key in vars(h.raw_contacts):
  if key.startswith('rigid_contact_'):setattr(h.contacts,key,getattr(h.raw_contacts,key))
 h.contacts.contact_generation=h.raw_contacts.contact_generation
base=np.r_[np.zeros(6),plan['base'],np.zeros(5)];press=np.r_[np.zeros(6),plan['press'][0],np.zeros(5)]
print('ACTIONS',h.action_names,flush=True)
sb=h.model.shape_body.numpy();keymap={b:k for k,b in enumerate(h.key_bodies)}
rows=[];cr=[];poses=[]
def collect(t):
 c=h.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);force=c.force.numpy()[:n,:3];a=sb[c.rigid_contact_shape0.numpy()[:n]];b=sb[c.rigid_contact_shape1.numpy()[:n]]
 for i in range(n):
  for own,other,sign in [(a[i],b[i],1.),(b[i],a[i],-1.)]:
   if other in keymap and 0<=own<h.hand_body_count and np.linalg.norm(force[i])>.001:
    cr.append([t,keymap[other]+1,'rigid',h.body_names[own],*(sign*force[i]).tolist(),float(np.linalg.norm(force[i]))])
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);sh=sb[c.soft_contact_shape.numpy()[:n]];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n];ff=h.out_f.numpy()[:n]
 for i in np.flatnonzero(np.isin(sh,h.key_bodies)&(np.linalg.norm(ff,axis=1)>.001)):
  reg=[]
  for j in range(3):
   if 0<=ii[i,j]<h.real_count and bb[i,j]>0:reg.append(h.region_names[h.regions[ii[i,j]]] if h.regions[ii[i,j]]>=0 else 'unmapped')
  cr.append([t,keymap[sh[i]]+1,'soft','+'.join(sorted(set(reg))),*ff[i].tolist(),float(np.linalg.norm(ff[i]))])

# Shape above the keys before vertical approach.
alpha=0.;lower=0.;finger_previous=-1
for i in range(650):
 t=i*.01
 if t<.5:
  target=base*(.5-.5*np.cos(np.pi*t/.5));phase='shape_above';code=0
 elif t<.8:target=base.copy();phase='settle_above';code=0
 elif t<1.3:
  if i==80:
   pq=h.state.particle_q.numpy();c=CENTERS[0];mask=(abs(pq[:,0]-c[0])<.009)&(abs(pq[:,1]-c[1])<.0075)&(np.arange(len(pq))<h.real_count)
   if not mask.any():print('ABORT no pad over key',flush=True);break
   lower=max(0.,min(.045,float(pq[mask,2].min())-.098-.003));print('LOWER',lower,flush=True)
  target=base.copy();target[1]=lower*(.5-.5*np.cos(np.pi*(t-.8)/.5));phase='lower_clear';code=0
 elif t<1.5:target=base.copy();target[1]=lower;phase='ready';code=0
 elif t<6.3:
  finger=min(3,int((t-1.5)/1.2));u=t-1.5-finger*1.2
  if finger!=finger_previous:alpha=0.;finger_previous=finger
  press=np.r_[np.zeros(6),plan['press'][finger],np.zeros(5)]
  if u<.75:
   prev=h.state.joint_q.numpy()[h.key_q][finger]
   if prev<.0025:alpha=min(1.,alpha+.015)
   elif prev>.003:alpha=max(0.,alpha-.01)
   phase=f'press_{finger+1}';code=1+finger*2
  elif u<1.:
   alpha=max(0.,alpha-.06);phase=f'release_{finger+1}';code=2+finger*2
  else:alpha=0.;phase=f'released_{finger+1}';code=2+finger*2
  target=base+(press-base)*alpha;target[1]=lower
 else:target=base.copy();target[1]=lower;phase='all_released';code=9
 # Separate wrist approach from little-finger pressing. All fingers are released first.
 if t>=4.9:
  extra=.002*(.5-.5*np.cos(np.pi*min((t-4.9)/.2,1.)))
  target=target.copy();target[1]=lower+extra
 h.audit_phase.assign(np.array([code],dtype=np.int32));h.set_targets(target);h.advance(.01,dt=.0000625 if t>=1.3 else .000125)
 travel=h.state.joint_q.numpy()[h.key_q];rows.append([h.time,phase,*travel.tolist()]);poses.append(h.state.body_q.numpy());collect(h.time)
 if i%10==0:print('FOUR',h.time,phase,alpha,travel*1000,flush=True)
 if not np.isfinite(travel).all() or np.max(np.abs(travel))>.01:print('ABORT divergent keys',flush=True);break
(out/(tag+'_substeps.json')).write_text(json.dumps(dict(phase_peaks_m=h.audit_peaks.numpy().tolist(),nonfinite_count=int(h.audit_bad.numpy()[0]),sample_dt=.0000625,phase_codes='0 approach;1/2 finger1 press/release;3/4 finger2;5/6 finger3;7/8 finger4;9 all released'),indent=2))
np.savez_compressed(out/(tag+'.npz'),time=np.array([r[0] for r in rows]),body_q=poses,body_names=h.body_names_all,key_travel=np.array([r[2:] for r in rows]))
for name,data,head in [(tag+'.csv',rows,['time','phase','key1_m','key2_m','key3_m','key4_m']),(tag+'_contacts.csv',cr,['time','key','type','body_or_region','fx_N','fy_N','fz_N','magnitude_N'])]:
 with (out/name).open('w') as f:w=csv.writer(f);w.writerow(head);w.writerows(data)
print('DONE',flush=True)
