
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json,numpy as np,mujoco as mj
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
ROOT=Path((str(_EVOLUTION_ROOT)));out=ROOT/'outputs/nov20/ax50_leaf04';out.mkdir(exist_ok=True)
m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
conv=json.load(open(ROOT/'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json'))['canonical_to_native'];sign=np.array([conv[m.joint(i).name.split('_Joints_')[-1]] for i in range(16)])
rot=R.from_euler('z',-15,degrees=True)*R.from_euler('x',180,degrees=True)*R.from_quat([0,.707106781,-.707106781,0]);ids=[12,16];points=np.array([[0,.005,.017],[0,.0044,.019]])
def tips(q,pos):
 d.qpos[:]=0;d.qpos[:16]=q;mj.mj_kinematics(m,d)
 return np.array([rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@point-[0,0,.3])+pos for b,point in zip(ids,points)])
# Under-leaf contact points at fixed lever arms, then hinged-leaf trajectory.
lever=np.array([.045,.053]);ys=[-.001,.020];zs=np.array([.080,.0808]);targets=[]
for deg in [0,20,40,52]:
 a=np.deg2rad(deg);targets.append(np.c_[-.067-lever*np.cos(a),ys,zs+lever*np.sin(a)-.002])
targets[0][:,2]-=.004
# Jointly select a shared wrist mount and per-stage index/middle angles.
q0=np.tile([0,.35,.2,.1],4)*sign;lo=m.jnt_range[:8,0]+1e-5;hi=m.jnt_range[:8,1]-1e-5
x0=np.r_[[-.005,-.01,.05],np.tile(q0[:8],4)]
def fun(x):
 rs=[]
 for k,tar in enumerate(targets):
  q=q0.copy();q[:8]=x[3+8*k:11+8*k];q[8:]=0;rs.extend(((tips(q,x[:3])-tar)*1000).ravel());rs.extend((q[:8]-q0[:8])*.05)
  angle=np.deg2rad([0,20,40,52][k]);normal=np.array([np.sin(angle),0,np.cos(angle)]);pad_normal=np.array([np.sin(angle+np.deg2rad(20)),0,np.cos(angle+np.deg2rad(20))])
  for b in ids:
   actual=rot.apply(d.xmat[b].reshape(3,3)@np.array([0,1,0]));rs.extend((actual-pad_normal)*8)

  for f,b in enumerate([11,15]):
   for zz in [0., [.024,.028][f]]:
    center=rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@np.array([0,0,zz])-[0,0,.3])+x[:3];gap=float((center-np.array([-.067,ys[f],zs[f]]))@normal)+[.0032,.0044][f]+.002
    rs.append(max(gap,0)*1500)
 rs.extend((x[:3]-[-.005,-.01,.05])*5);return np.array(rs)
f=least_squares(fun,x0,bounds=(np.r_[[-.035,-.05,.025],np.tile(lo,4)],np.r_[[.025,.025,.085],np.tile(hi,4)]),max_nfev=1500)
poses=[];errors=[]
for k,tar in enumerate(targets):
 q=np.zeros(16);q[:8]=f.x[3+8*k:11+8*k];poses.append((q*sign).tolist());errors.append(((tips(q,f.x[:3])-tar)*1000).tolist())
r=dict(position=f.x[:3].tolist(),orientation=rot.as_quat().tolist(),angles_deg=[0,20,40,52],poses=poses,targets=np.array(targets).tolist(),errors_mm=errors,tip_local=points.tolist(),status='CPU FK only; physical full-surface trajectory not validated')
(out/'plan_tipclear.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
