
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
place(.07815,.04675);r.collision.collide(r.state,c);eval_rigid_contact_kinematics(r.model,r.state,c,out_distance=dist)
n=int(c.rigid_contact_count.numpy()[0]);a=c.rigid_contact_shape0.numpy()[:n];b=c.rigid_contact_shape1.numpy()[:n];dd=dist.numpy()[:n];i=min([i for i in range(n) if a[i]>=0 and b[i]>=0 and 'link_1_2' in labels[a[i]] and labels[b[i]]=='brush_shaft'],key=lambda i:dd[i]);bi=sb[a[i]];point=c.rigid_contact_point0.numpy()[i];pose=r.state.body_q.numpy()[bi];nl=Rot.from_quat(pose[3:]).inv().apply(c.rigid_contact_normal.numpy()[i]);q0=r.state.joint_q.numpy().copy();names=['thumb_sweep','arch_1','thumb_axial','digit_1_flex_1','digit_1_flex_2'];ids=[r.joint_names.index(n) for n in names];lo=r.model.joint_limit_lower.numpy()[ids];hi=r.model.joint_limit_upper.numpy()[ids]
def residual(v):
 q=q0.copy();q[ids]=v;r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);p=r.state.body_q.numpy()[bi];R=Rot.from_quat(p[3:]);w=R.apply(point)+p[:3];nn=R.apply(nl)
 return np.r_[(w[:2]-[.07315,.04675])*1000,(nn-[1,0,0])*10,max(0,.095-w[2],w[2]-.21)*1000,(v-q0[ids])*.05]
best=None
for k in range(6):
 initial=q0[ids] if k==0 else np.random.default_rng(k).uniform(lo,hi)
 fit=least_squares(residual,np.clip(initial,lo+1e-6,hi-1e-6),bounds=(lo,hi),diff_step=1e-3,max_nfev=180)
 score=float(np.linalg.norm(fit.fun));print('FIT',k,score,flush=True)
 if best is None or score<best[0]:best=(score,fit.x.copy())
residual(best[1]);thumbfit=dict(zip(names,best[1].tolist()));(O/'audit22_thumbfit.json').write_text(json.dumps({'score':best[0],'targets':thumbfit,'point_body':point.tolist(),'normal_body':nl.tolist()},indent=2));print('THUMBFIT',best,thumbfit,flush=True)
