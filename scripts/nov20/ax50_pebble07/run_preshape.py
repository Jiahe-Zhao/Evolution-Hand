"""Sequential thumb/index preshape away from objects; never task acceptance."""
from pebble_runtime import *
import csv,os
TAG=os.environ.get("RUN_TAG","preshape_fixed")
h=PebbleRuntime(side='left',wrist='six_axis',position=(.10,.10,.20),gravity=(0,0,-9.81))
initial=h.state.body_q.numpy()[h.object_bodies,:3].copy(); rows=[];bodies=[];joints=[];particles=[];targets=[];stop='completed';maximum=np.zeros(3)
smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
with (OUT/f'{TAG}_trace.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['time_s','phase','finite','soft_max_speed_m_s','target_displacement_mm','neighbor1_displacement_mm','neighbor2_displacement_mm','hand_joint_max_error_rad'])
 for i in range(90):
  t=i*.01;u=smooth(t/.3);v=smooth((t-.3)/.3);target=np.zeros(len(h.action_names));phase='thumb_preshape' if t<.3 else 'index_preshape' if t<.6 else 'hold'
  for name,value in [('thumb_sweep',.35),('arch_1',.10),('thumb_axial',.25),('digit_1_flex_1',.3),('digit_1_flex_2',.25)]:target[h.action_names.index(name)]=value*u
  for name,value in [('digit_2_flex_1',.2),('digit_2_flex_2',.3),('digit_2_flex_3',.2)]:target[h.action_names.index(name)]=value*v
  h.set_targets(target);h.advance(.01,dt=.000125)
  b=h.state.body_q.numpy();q=h.state.joint_q.numpy();p=h.state.particle_q.numpy();vel=h.state.particle_qd.numpy();finite=all(np.isfinite(x).all() for x in [b,q,p,vel,h.state.joint_qd.numpy()]);speed=float(np.max(np.linalg.norm(vel,axis=1)));disp=np.linalg.norm(b[h.object_bodies,:3]-initial,axis=1)*1000;maximum=np.maximum(maximum,disp)
  error=float(np.max(np.abs(q[h.action_indices[6:]]-target[6:])));row=[h.time,phase,finite,speed,*disp.tolist(),error];rows.append(row);w.writerow(row);f.flush();bodies.append(b);joints.append(q);particles.append(p[:h.real_count]);targets.append(target)
  if i%10==0:print('STEP',row,flush=True)
  if not finite or speed>20:stop='nonfinite_or_runaway';break
np.savez_compressed(OUT/f'{TAG}.npz',time=[r[0] for r in rows],body_q=bodies,joint_q=joints,particles=particles,targets=targets,body_names=h.model.body_label,tri=h.model.tri_indices.numpy())
result=dict(scope='sequential thumb then index preshape away from objects; no pinch or task acceptance',stop=stop,time=h.time,max_object_displacement_mm=maximum.tolist(),max_soft_speed_m_s=max(r[3] for r in rows),final_hand_joint_max_error_rad=rows[-1][-1],action_names=h.action_names,finite=all(r[2] for r in rows))
(OUT/f'{TAG}_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
