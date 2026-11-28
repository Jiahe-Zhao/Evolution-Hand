
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from keys_runtime import *
import json,sys,csv
out=ROOT/'outputs/nov20/ax50_keys17';mode=sys.argv[1];pose=sys.argv[2];plan=json.loads((out/'left_plan.json').read_text())
import palm_adapter as da
from wrist_runtime import wrist_pd
da.control_pd_subset=wrist_pd
Parent=da.AX50BrushRuntime
class Isolated(Parent):
 def _add_brush(self,b):self.prop_body_count=0
h=Isolated(side='left',wrist='six_axis' if mode=='wrist' else 'locked',position=(np.array(plan['position'])+[0,0,.04]).tolist(),orientation=plan['orientation'],gravity=(0,0,-9.81))
base=np.r_[plan['base'],np.zeros(5)];base=np.r_[np.zeros(6),base] if mode=='wrist' else base
rows=[];result='finite'
for i in range(100):
 t=i*.01;ratio=(.5-.5*np.cos(np.pi*min(t/.5,1))) if pose=='pose' else 0.
 h.set_targets(base*ratio);h.advance(.01,dt=.000125)
 q=h.state.joint_q.numpy();qd=h.state.joint_qd.numpy();p=h.state.particle_q.numpy();v=h.state.particle_qd.numpy()
 finite=bool(all(np.isfinite(x).all() for x in [q,qd,p,v]))
 row=[h.time,finite,float(np.nanmax(abs(qd))),float(np.nanmax(np.linalg.norm(v,axis=1))),float(np.nanmax(np.linalg.norm(p,axis=1)))]
 rows.append(row)
 if i%10==0:print(mode,pose,row,flush=True)
 if not finite:result='nonfinite';break
 if row[2]>1000 or row[3]>100:result='velocity_runaway';break
import os
stem='palm_skin5_ablation'
with (out/(stem+'.csv')).open('w') as f:w=csv.writer(f);w.writerow(['time_s','finite','max_joint_speed','max_particle_speed_m_s','max_particle_radius_m']);w.writerows(rows)
(out/(stem+'.json')).write_text(json.dumps(dict(mode=mode,pose=pose,result=result,last=rows[-1],objects='none',dt=.000125,filtered_contacts=int(h.attachment_removed.numpy()[0])),indent=2));print('RESULT',result,rows[-1],flush=True)
