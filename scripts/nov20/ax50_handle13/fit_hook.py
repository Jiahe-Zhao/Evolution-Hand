
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/ax50_handle13';m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
conv=json.load(open(ROOT/'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json'))['canonical_to_native'];sign=np.array([conv[m.joint(i).name.split('_Joints_')[-1]] for i in range(16)])
rot0=R.from_euler('x',180,degrees=True)*R.from_quat([0,.707106781,-.707106781,0]);rot=rot0;ids=[15,19];goal=np.array([[-.125,-.037,.044],[-.125,-.012,.044]])
def pose(x):
 q=np.array([0.,.8,1.,.8,0,0,0,0,0,0,0,0,0,.6,1.4,1.]);q[4:8]=[x[3],x[4],1.4,1.0];q[8:12]=[x[5],x[6],1.4,1.0];return q

def fk(q,pos):
 d.qpos[:]=0;d.qpos[:16]=q*sign;mj.mj_kinematics(m,d)
 return np.array([rot.apply(d.xpos[b]-[0,0,.3])+pos for b in ids])
def fun(x):
 global rot
 rot=R.from_euler('z',x[7])*rot0
 q=pose(x);return np.r_[((fk(q,x[:3])-goal)*1000).ravel(), (x[[3,5]])*4,(x[[4,6]]-.15)*2]
x=least_squares(fun,[-.01,-.05,.035,0,.15,0,.15,.15],bounds=([-.06,-.1,.015,-.15,.001,-.15,.001,-.52],[.05,0,.06,.5,.7,.5,.7,.52])).x;q=pose(x);rot=R.from_euler('z',x[7])*rot0
r=dict(position=x[:3].tolist(),orientation=rot.as_quat().tolist(),hook_pose=q.tolist(),pip_target_m=goal.tolist(),pip_actual_m=fk(q,x[:3]).tolist(),scope='CPU FK candidate only; not contact or task validation',hook_fingers=['middle','ring']);(OUT/'plan.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
