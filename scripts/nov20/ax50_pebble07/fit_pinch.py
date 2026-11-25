"""CPU thumb/index opposition candidate; does not certify collision clearance."""
from pathlib import Path
import numpy as np,json,mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_pebble07'
m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith(n)) for n in ['link_1_2','link_2_3']];slots=np.array([0,1,2,3,16,17,18,19,20]);points=[np.array([0,.004,.014]),np.array([0,.0045,.016])]
lo=m.jnt_range[slots,0]+1e-5;hi=m.jnt_range[slots,1]-1e-5
seed=np.array([.05,.6,.8,.5,.6,.2,.3,.6,.5])
def fk(x):
 d.qpos[:]=0;d.qpos[slots]=x;mj.mj_kinematics(m,d)
 p=np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@v for b,v in zip(ids,points)]);n=np.array([d.xmat[b].reshape(3,3)@np.array([0,1,0]) for b in ids]);return p,n

def err(x):
 p,n=fk(x);delta=p[1]-p[0];length=np.linalg.norm(delta);axis=delta/length
 return np.r_[(length-.020)*1000,(n[0]-axis)*10,(n[1]+axis)*10,(x-seed)*.1]
r=least_squares(err,seed,bounds=(lo,hi),max_nfev=500)
p,n=fk(r.x);axis=(p[1]-p[0])/np.linalg.norm(p[1]-p[0]);approach=p.mean(0)-np.array([0,0,.3]);approach-=axis*(axis@approach);approach/=np.linalg.norm(approach);rot,_=R.align_vectors(np.array([[0,1,0],[0,0,-1]]),np.array([axis,approach]));center=np.array([-.093,.003,.035]);mount=center-rot.apply(p.mean(0)-[0,0,.3]);q=np.zeros(21);q[slots]=r.x
result=dict(scope='CPU FK candidate only; local pad points approximate, no collision or grasp validation',terminal_bodies=['link_1_2','link_2_3'],joint_pose=q.tolist(),position=mount.tolist(),orientation=rot.as_quat().tolist(),gap_mm=float(np.linalg.norm(p[1]-p[0])*1000),normal_alignment=[float(n[0]@axis),float(-n[1]@axis)],local_points=[v.tolist() for v in points],solver_success=bool(r.success))
(OUT/'pinch_candidate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
