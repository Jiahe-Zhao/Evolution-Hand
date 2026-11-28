
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np,mujoco as mj,json
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
out=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_keys17'));m=mj.MjModel.from_binary_path(str(out/'left_ik.mjb'));d=mj.MjData(m)
rot=R.from_matrix(np.array([[-1,0,0],[0,0,-1],[0,-1,0]]));ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith(f'link_{k}_3')) for k in range(2,6)]
centers=np.array([[-.123,.001,.102],[-.128,-.018,.102],[-.118,-.037,.102],[-.097,-.056,.102]])
points=np.array([[0,-.005,.012],[0,-.004,.014],[0,-.004,.013],[0,-.004,.012]])
def tips(q,pos):
 d.qpos[:]=0;d.qpos[:16]=q;mj.mj_kinematics(m,d)
 return np.array([rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@p-[0,0,.3])+pos for b,p in zip(ids,points)])
lo=np.r_[[-.04,-.06,.105],m.jnt_range[:16,0]+1e-5];hi=np.r_[[.04,.03,.17],m.jnt_range[:16,1]-1e-5]
signs=np.array([json.loads((out.parents[1]/'ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json').read_text())['canonical_to_native'][m.joint(i).name.split('_Joints_')[-1]] for i in range(16)])
q0=np.tile([.05,.2,.2,.1],4)*signs
def fun(v):return np.r_[(tips(v[3:],v[:3])-centers).ravel()*1000,(v[3:]-q0)*.15,(v[:3]-[0,-.02,.13])*5]
f=least_squares(fun,np.r_[[0,-.02,.13],q0],bounds=(lo,hi),max_nfev=1500)
base=f.x[3:];pos=f.x[:3];press=[]
for k in range(4):
 target=centers[k].copy();target[2]=.095
 def pressfun(v):
  q=base.copy();q[k*4:k*4+4]=v
  return np.r_[(tips(q,pos)[k]-target)*1000,(v-base[k*4:k*4+4])*.1]
 ff=least_squares(pressfun,base[k*4:k*4+4],bounds=(lo[3+k*4:7+k*4],hi[3+k*4:7+k*4]),max_nfev=500)
 q=base.copy();q[k*4:k*4+4]=ff.x;press.append(q.tolist())
r=dict(position=pos.tolist(),orientation=rot.as_quat().tolist(),base=(base*signs).tolist(),press=[(np.array(p)*signs).tolist() for p in press],tip_local=points.tolist(),rest_tips=tips(base,pos).tolist(),fit_error_mm=((tips(base,pos)-centers)*1000).tolist(),note='CPU FK planning only, no physical validation; left hand')
(out/'left_plan.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
