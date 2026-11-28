
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,mujoco as mj,json
from scipy.optimize import least_squares
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_brush'));m=mj.MjModel.from_binary_path(str(P/'hand_ik.mjb'));d=mj.MjData(m)
names=[m.joint(i).name.split('_Joints_')[-1] for i in range(m.njnt)];bi=[]
for n in ['link_1_2','link_2_3','link_3_3','link_4_3']:
 bi.append(next(i for i in range(m.nbody) if m.body(i).name.endswith('_'+n)))
ix=np.arange(6,27);lo=m.jnt_range[ix,0];hi=m.jnt_range[ix,1];guess=np.array([.25 if names[i].startswith('arch') else .7 for i in ix]);guess[names[6:27].index('thumb_sweep')]=.1;guess[names[6:27].index('thumb_axial')]=.8
signs=np.array([[-1,0],[1,0],[1,0],[.5, .866]])
center=np.array([.07,.042]);local=np.array([0,.006,.011]);best=None
for k in range(5):
 def residual(v):
  d.qpos[:]=0;d.qpos[ix]=v[:21];mj.mj_forward(m,d);c=v[21:23];res=[]
  for j,b in enumerate(bi):
   R=d.xmat[b].reshape(3,3);p=d.xpos[b]+R@local;n=R[:,1];rad=signs[j]
   res.extend((p[:2]-c-rad*.0052)*1000)
   res.extend((n[:2]+rad)*2)
  res.extend((v[:21]-guess)*.2);res.extend((c-center)*50)
  # Reject existing rigid self penetration rather than hide it in the display.
  depths=np.zeros(8)
  neg=sorted([-x.dist for x in d.contact if x.dist<-.0005],reverse=True)[:8]
  depths[:len(neg)]=np.array(neg)*300
  res.extend(depths)
  return res
 x=np.r_[np.clip(guess+np.random.default_rng(k).normal(0,.15,21),lo+.001,hi-.001),center]
 r=least_squares(residual,x,bounds=(np.r_[lo,[.035,.015]],np.r_[hi,[.10,.075]]),max_nfev=180,diff_step=1e-4)
 if best is None or np.linalg.norm(r.fun)<best[0]:best=(np.linalg.norm(r.fun),r.x)
 print(k,np.linalg.norm(r.fun),flush=True)
v=best[1];residual(v);out={'native_q':d.qpos.tolist(),'joint_names':names,'rod_xy':v[21:23].tolist(),'fit_residual_norm':best[0],'status':'kinematic candidate only; no grasp validation','tip_positions':[(d.xpos[b]+d.xmat[b].reshape(3,3)@local).tolist() for b in bi]};(P/'grip_candidate.json').write_text(json.dumps(out,indent=2));print(out)
