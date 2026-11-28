
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from keys_runtime import *
import json,sys,csv
out=ROOT/'outputs/nov20/ax50_keys17';mode='wrist';pose='sequential';plan=json.loads((out/'left_plan.json').read_text())
if mode=='wrist':
 from wrist_runtime import WristKeysRuntime
 Parent=WristKeysRuntime
else:Parent=KeysRuntime
class Isolated(Parent):
 def _add_brush(self,b):self.prop_body_count=0
h=Isolated(side='left',wrist='six_axis' if mode=='wrist' else 'locked',position=(np.array(plan['position'])+[0,0,.04]).tolist(),orientation=plan['orientation'],gravity=(0,0,-9.81))
base=np.r_[plan['base'],np.zeros(5)];base=np.r_[np.zeros(6),base] if mode=='wrist' else base
rows=[];result='finite'
stable_count=0;wrist_started=None
for i in range(350):
 t=i*.01;target=np.zeros_like(base)
 for finger in range(4):
  r=np.clip((t-finger*.4)/.4,0,1);target[6+finger*4:10+finger*4]=base[6+finger*4:10+finger*4]*(.5-.5*np.cos(np.pi*r))
 phase='fingers' if t<1.6 else 'settle'
 if t>=1.6:
  qd0=h.state.joint_qd.numpy();v0=h.state.particle_qd.numpy()
  if np.isfinite(qd0).all() and np.isfinite(v0).all() and np.max(abs(qd0[6:22]))<.1 and np.max(np.linalg.norm(v0,axis=1))<.05:stable_count+=1
  else:stable_count=0
  if stable_count>=20 and wrist_started is None:wrist_started=t
 if wrist_started is not None:
  r=np.clip((t-wrist_started)/.5,0,1);target[1]=.005*(.5-.5*np.cos(np.pi*r));phase='wrist_after_settle'
 h.set_targets(target);h.advance(.01,dt=.000125)
 q=h.state.joint_q.numpy();qd=h.state.joint_qd.numpy();p=h.state.particle_q.numpy();v=h.state.particle_qd.numpy()
 finite=bool(all(np.isfinite(x).all() for x in [q,qd,p,v]))
 row=[h.time,finite,float(np.nanmax(abs(qd))),float(np.nanmax(np.linalg.norm(v,axis=1))),float(np.nanmax(np.linalg.norm(p,axis=1)))]
 rows.append(row+[phase,*q[:6].tolist(),*qd[:6].tolist()])
 if i%10==0:print(mode,pose,row,flush=True)
 if not finite:result='nonfinite';break
 if row[2]>1000 or row[3]>100:result='velocity_runaway';break
stem='isolate_'+mode+'_'+pose
with (out/(stem+'.csv')).open('w') as f:w=csv.writer(f);w.writerow(['time_s','finite','max_joint_speed','max_particle_speed_m_s','max_particle_radius_m','phase',*[f'wrist_q{k}' for k in range(6)],*[f'wrist_qd{k}' for k in range(6)]]);w.writerows(rows)
(out/(stem+'.json')).write_text(json.dumps(dict(mode=mode,pose=pose,result=result,last=rows[-1],objects='none',dt=.000125,wrist_started=wrist_started,settle_requirement='20 consecutive frames with finger speed<0.1rad/s and soft speed<0.05m/s'),indent=2));print('RESULT',result,rows[-1],flush=True)
