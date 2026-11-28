"""Fixed-wrist independent terminal approach paths, millimetres along measured nearest-surface directions."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
P=Path(__file__).parent;ROOT=Path((str(_EVOLUTION_ROOT)))
c=json.loads((P/'candidates/pinch_candidate_v4.json').read_text());close=json.loads((P/'candidates/pinch_close_candidate_v4.json').read_text())
m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m);rot=R.from_quat(c['orientation']);base=np.array(c['position'])+close['wrist_delta_world'];q=np.zeros(m.nq)
slots=[next(i for i in range(m.njnt) if m.joint(i).name.endswith('_Joints_'+n)) for n in close['joint_names']]
for j,v in zip(slots,close['joint_values']):q[m.jnt_qposadr[j]]=v
out=dict(scope='CPU approach paths only; force feedback required in actual simulation',travel_mm=np.linspace(0,2.5,51).tolist(),fingers={})
for key,term,local,indices,sign in [('thumb','link_1_2',c['local_points'][0],range(4,9),1),('index','link_2_3',c['local_points'][1],range(4),-1)]:
 js=np.array([slots[i] for i in indices]);addr=m.jnt_qposadr[js];bid=next(i for i in range(m.nbody) if m.body(i).name.endswith(term));seed=q[addr].copy();x=seed.copy()
 def fk(v):
  d.qpos[:]=q;d.qpos[addr]=v;mj.mj_kinematics(m,d)
  return rot.apply(d.xpos[bid]+d.xmat[bid].reshape(3,3)@local-[0,0,.3])+base
 start=fk(seed);poses=[];err=[]
 for mm in out['travel_mm']:
  direction=np.array([.1677,.8954,-.4125] if key=='thumb' else [-.0307,-.9517,-.3056]);direction/=np.linalg.norm(direction)
  goal=start+direction*mm/1000
  res=least_squares(lambda v:np.r_[(fk(v)-goal)*1000,(v-seed)*.01],x,bounds=(m.jnt_range[js,0]+.03,m.jnt_range[js,1]-.03),max_nfev=150)
  x=res.x;poses.append(x.tolist());err.append(float(np.linalg.norm(fk(x)-goal)*1000))
 out['fingers'][key]=dict(names=[close['joint_names'][i] for i in indices],joint_values=poses,max_error_mm=max(err),start_world=start.tolist())
 print(key,[(round(mm,2),round(e,3)) for mm,e in zip(out['travel_mm'],err) if e>.05][:5]);assert max(err)<.5,(key,max(err))
(P/'candidates/contact_search_v5.json').write_text(json.dumps(out,indent=2));print({k:v['max_error_mm'] for k,v in out['fingers'].items()})
