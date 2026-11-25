from keys_runtime import *
import json,csv
out=ROOT/'outputs/nov20/ax50_keys17';plan=json.loads((out/'coordinator_plan.json').read_text())
h=KeysRuntime(side='right',wrist='locked',position=plan['position'],orientation=plan['orientation'],gravity=(0,0,-9.81))
base=np.r_[plan['base'],np.zeros(5)];press=np.r_[plan['press'][0],np.zeros(5)]
sb=h.model.shape_body.numpy();keymap={b:k for k,b in enumerate(h.key_bodies)}
rows=[];cr=[];poses=[]
def collect(t):
 c=h.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);force=c.force.numpy()[:n,:3];a=sb[c.rigid_contact_shape0.numpy()[:n]];b=sb[c.rigid_contact_shape1.numpy()[:n]]
 for i in range(n):
  for own,other in [(a[i],b[i]),(b[i],a[i])]:
   if other in keymap and 0<=own<h.hand_body_count and np.linalg.norm(force[i])>.001:
    cr.append([t,keymap[other]+1,'rigid',h.body_names[own],*force[i].tolist(),float(np.linalg.norm(force[i]))])
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);sh=sb[c.soft_contact_shape.numpy()[:n]];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n];ff=h.out_f.numpy()[:n]
 for i in np.flatnonzero(np.isin(sh,h.key_bodies)&(np.linalg.norm(ff,axis=1)>.001)):
  reg=[]
  for j in range(3):
   if 0<=ii[i,j]<h.real_count and bb[i,j]>0:reg.append(h.region_names[h.regions[ii[i,j]]] if h.regions[ii[i,j]]>=0 else 'unmapped')
  cr.append([t,keymap[sh[i]]+1,'soft','+'.join(sorted(set(reg))),*ff[i].tolist(),float(np.linalg.norm(ff[i]))])
for i in range(140):
 t=i*.01
 if t<.4:target=base*(.5-.5*np.cos(np.pi*t/.4));phase='approach'
 elif t<.6:target=base;phase='hover'
 elif t<.9:target=base+(press-base)*(.5-.5*np.cos(np.pi*(t-.6)/.3));phase='press'
 elif t<1.05:target=press;phase='hold'
 elif t<1.3:target=press+(base-press)*(.5-.5*np.cos(np.pi*(t-1.05)/.25));phase='release'
 else:target=base;phase='released'
 h.set_targets(target);h.advance(.01)
 travel=h.state.joint_q.numpy()[h.key_q];rows.append([h.time,phase,*travel.tolist()]);poses.append(h.state.body_q.numpy());collect(h.time)
 if i%10==0:print('SINGLE',h.time,phase,travel*1000,flush=True)
np.savez_compressed(out/'single_coordinator.npz',time=np.array([r[0] for r in rows]),body_q=poses,body_names=h.body_names_all,key_travel=np.array([r[2:] for r in rows]))
for name,data,head in [('single_coordinator.csv',rows,['time','phase','key1_m','key2_m','key3_m','key4_m']),('single_coordinator_contacts.csv',cr,['time','key','type','body_or_region','fx_N','fy_N','fz_N','magnitude_N'])]:
 with (out/name).open('w') as f:w=csv.writer(f);w.writerow(head);w.writerows(data)
print('DONE',flush=True)
