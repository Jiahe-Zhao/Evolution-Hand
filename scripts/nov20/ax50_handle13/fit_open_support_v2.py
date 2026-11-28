
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,mujoco as mj,json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/ax50_handle13';m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
conv=json.loads((ROOT/'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json').read_text())['canonical_to_native'];sign=np.array([conv[m.joint(i).name.split('_Joints_')[-1]] for i in range(16)])
ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith('_'+n)) for n in ['link_3_2','link_4_2']];points=np.array([[0,.0044,.012],[0,.0044,.012]]);goals=np.array([[-.112,-.035,.047],[-.112,-.010,.047]]);rot0=R.from_euler('x',180,degrees=True)*R.from_quat([0,.707106781,-.707106781,0])
def evaluate(x):
 q=np.zeros(16);q[4]=x[3];q[8]=x[4];d.qpos[:]=0;d.qpos[:16]=q*sign;mj.mj_kinematics(m,d);rot=R.from_euler('z',x[5])*rot0
 tip=np.array([rot.apply(d.xpos[i]+d.xmat[i].reshape(3,3)@v-[0,0,.3])+x[:3] for i,v in zip(ids,points)]);return q,rot,tip
def residual(x):
 q,rot,tip=evaluate(x);return np.r_[((tip-goals)*1000).ravel(),x[3:5]*4]
# Keep the wrist square to the crossbar; allow different longitudinal contact
# locations on the middle and ring middle phalanges instead of yawing the hand.
x=np.array([0.,0.,0.,0.,0.,0.]);q,rot,tip=evaluate(x)
pos=np.array([-.112,-.022,.047])-tip.mean(0);x[:3]=pos;q,rot,tip=evaluate(x)
contact=[]
for b in ids:
 origin=rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([0,.0044,0])-[0,0,.3])+pos
 axis=rot.apply(d.xmat[b].reshape(3,3)@np.array([0,0,1]))
 local_z=(-.112-origin[0])/axis[0];contact.append(dict(body=m.body(b).name,local_point=[0,.0044,float(local_z)],world_point=(origin+local_z*axis).tolist()))
r=dict(position=pos.tolist(),orientation=rot.as_quat().tolist(),open_pose=q.tolist(),contact_points=contact,scope='CPU FK candidate only, square wrist, MCP/PIP/DIP zero, no physical pass')
(OUT/'open_support_plan_v2.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
