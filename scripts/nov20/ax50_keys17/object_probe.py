from keys_runtime import *
import csv,json
out=ROOT/'outputs/nov20/ax50_keys17'
h=KeysRuntime(side='right',wrist='locked',position=(0,0,.3),gravity=(0,0,-9.81))
rows=[]
for i in range(240):
 t=i*.01;key=int(t//.6);phase=t% .6
 force=np.zeros(h.model.joint_dof_count)
 if .1<=phase<.3:force[h.key_dof[key]]=.72
 h.control.joint_f.assign(force)
 h.advance(.01)
 rows.append([h.time,key,*h.state.joint_q.numpy()[h.key_q].tolist()])
 if i%30==0:print('PROBE',rows[-1],flush=True)
a=np.array(rows)
with (out/'object_probe.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['time_s','actuated_key_zero_based','key1_m','key2_m','key3_m','key4_m']);w.writerows(rows)
r=[]
for k in range(4):
 v=a[a[:,1]==k,2:];others=np.delete(v,k,axis=1)
 r.append(dict(key=k+1,peak_mm=float(v[:,k].max()*1000),neighbor_peak_mm=float(others.max()*1000),release_mm=float(v[-1,k]*1000),pass_check=bool(v[:,k].max()>=.002 and others.max()<.002 and v[-1,k]<=.0005)))
(out/'object_probe.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
