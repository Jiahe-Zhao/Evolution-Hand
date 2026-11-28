
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
exec((ROOT/'scripts/nov20/ax50_brush/audit22_thumbfit.py').read_text().split('best=None')[0])
tree=cKDTree(basep[ix,:2]);start=np.r_[q0[ids],[.07815,.04675]];low=np.r_[lo,[.070,.040]];high=np.r_[hi,[.09,.065]]
def free_residual(v):
 q=q0.copy();q[ids]=v[:5];r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);p=r.state.body_q.numpy()[bi];R=Rot.from_quat(p[3:]);w=R.apply(point)+p[:3];nt=R.apply(nl);center=v[5:];distance,k=tree.query(center);ni=np.r_[(center-basep[ix[k],:2])/max(distance,1e-9),0.]
 return np.r_[(w[:2]+.005*nt[:2]-center)*1000,(nt+ni)*15,(distance-.00505)*1000,max(0,.07-w[2],w[2]-.21)*1000,(v[:5]-q0[ids])*.02]
bestfree=None
for k in range(8):
 initial=start.copy() if k==0 else np.r_[np.random.default_rng(k).uniform(lo,hi),[.078,.052]]
 fit=least_squares(free_residual,np.clip(initial,low+1e-6,high-1e-6),bounds=(low,high),diff_step=1e-3,max_nfev=200)
 score=float(np.linalg.norm(fit.fun));print('FREE_FIT',k,score,flush=True)
 if bestfree is None or score<bestfree[0]:bestfree=(score,fit.x.copy())
print('BESTFREE',bestfree,free_residual(bestfree[1]),flush=True);(O/'audit22_freefit.json').write_text(json.dumps({'score':bestfree[0],'targets':dict(zip(names,bestfree[1][:5].tolist())),'center':bestfree[1][5:].tolist()},indent=2))
