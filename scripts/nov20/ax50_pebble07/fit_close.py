from pathlib import Path
import numpy as np,json,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_pebble07';c=json.loads((OUT/'pinch_candidate.json').read_text());m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
slots=np.array([next(i for i in range(m.njnt) if m.joint(i).name.endswith('_Joints_'+n)) for n in c['joint_names']]);ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith(n)) for n in c['terminal_bodies']];rot=R.from_quat(c['orientation']);base=np.array(c['position']);pts=np.array(c['local_points']);seed=np.array(c['joint_values']);goals=np.array([[-.093,.003-.00675,.0345],[-.093,.003+.00675,.0345]])
def fk(x):
 d.qpos[:]=0;d.qpos[slots]=x[:9];mj.mj_kinematics(m,d)
 return np.array([rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@v-[0,0,.3])+base+x[9:] for b,v in zip(ids,pts)])
def residual(x):return np.r_[((fk(x)-goals)*1000).ravel(),(x[:9]-seed)*.2,x[9:]*20]
lo=np.r_[m.jnt_range[slots,0]+.05,[-.02]*3];hi=np.r_[m.jnt_range[slots,1]-.05,[.02]*3];res=least_squares(residual,np.r_[seed,[0,0,0]],bounds=(lo,hi),max_nfev=500);err=np.linalg.norm(fk(res.x)-goals,axis=1)*1000
r=dict(scope='CPU closure candidate; no physical result',joint_names=c['joint_names'],joint_values=res.x[:9].tolist(),wrist_delta_world=res.x[9:].tolist(),tip_error_mm=err.tolist(),min_joint_margin_rad=float(np.minimum(res.x[:9]-m.jnt_range[slots,0],m.jnt_range[slots,1]-res.x[:9]).min()),accepted_for_trial=bool(err.max()<.5))
(OUT/'pinch_close_candidate.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
