
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np,json,mujoco as mj
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_brush'));O=P/'continue';m=mj.MjModel.from_binary_path(str(P/'hand_ik.mjb'));d=mj.MjData(m);z=np.load(O/'audit22_pinch_prepared.npz');j=[m.joint(i).name.split('_Joints_')[-1] for i in range(m.njnt)];q=np.r_[np.zeros(6),z['joint_q']];d.qpos[:]=q;mj.mj_kinematics(m,d);b=next(i for i in range(m.nbody) if m.body(i).name.endswith('_link_1_2'));old=json.loads((O/'audit22_thumbfit.json').read_text());point=np.array(old['point_body']);nl=np.array(old['normal_body']);mount=np.array([0,0,.18]);print('PATCH',d.xpos[b]+d.xmat[b].reshape(3,3)@point+mount,flush=True)
pads=np.load((str(_EVOLUTION_ROOT) + '/outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/pads.npz'));palm=np.load((str(_EVOLUTION_ROOT) + '/outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/membrane.npz'));pl=(pads['vertices']-pads['rest_body_matrices'][0,:3,3])@pads['rest_body_matrices'][0,:3,:3];offset=len(palm['surface_rest_vertices_m']);ix=np.flatnonzero(pl[:,2]>.063)+offset;pts=z['particle_q'][ix];tree=cKDTree(pts[:,:2]);names=['thumb_sweep','arch_1','thumb_axial','digit_1_flex_1','digit_1_flex_2'];ids=[j.index(n) for n in names];lo=np.r_[m.jnt_range[ids,0],[.07,.04]];hi=np.r_[m.jnt_range[ids,1],[.095,.07]];f0=json.loads((O/'audit22_freefit.json').read_text());start=np.r_[q[ids],f0['center']]
def residual(v):
 d.qpos[:]=q;d.qpos[ids]=v[:5];mj.mj_kinematics(m,d);w=d.xpos[b]+d.xmat[b].reshape(3,3)@point+mount;nt=d.xmat[b].reshape(3,3)@nl;center=v[5:];dist,k=tree.query(center);ni=np.r_[(center-pts[k,:2])/max(dist,1e-9),0.]
 return np.r_[(w[:2]+.005*nt[:2]-center)*1000,(nt+ni)*15,(dist-.00505)*1000,(w[2]-pts[k,2])*1000,(v[:5]-q[ids])*.02]
best=None
for k in range(12):
 initial=start.copy() if k==0 else np.r_[np.random.default_rng(k).uniform(lo[:5],hi[:5]),[.0825,.063]]
 fit=least_squares(residual,np.clip(initial,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=500,diff_step=1e-5);score=float(np.linalg.norm(fit.fun));print('FIT',k,score,flush=True)
 if best is None or score<best[0]:best=(score,fit.x.copy())
print('BEST',best,residual(best[1]),flush=True);(O/'audit22_axialfit.json').write_text(json.dumps({'score':best[0],'targets':dict(zip(names,best[1][:5].tolist())),'center':best[1][5:].tolist(),'residual':residual(best[1]).tolist()},indent=2))
