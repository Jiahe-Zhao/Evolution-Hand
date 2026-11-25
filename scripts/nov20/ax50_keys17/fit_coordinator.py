import numpy as np,mujoco as mj,json
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
out=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_keys17');m=mj.MjModel.from_binary_path(str(out/'keys_ik.mjb'));d=mj.MjData(m)
rot=R.from_euler('x',-90,degrees=True);ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith(f'link_{k}_3')) for k in range(2,6)]
centers=np.array([[-.123,.001,.102],[-.128,-.018,.102],[-.118,-.037,.102],[-.097,-.056,.102]])
points=np.array([[0,-.005,.012],[0,-.004,.014],[0,-.004,.013],[0,-.004,.012]])
def tips(q,pos):
 d.qpos[:]=0;d.qpos[:16]=q;mj.mj_kinematics(m,d)
 return np.array([rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@p-[0,0,.3])+pos for b,p in zip(ids,points)])
lo=np.r_[[-.30,-.06,.12],m.jnt_range[:16,0]+1e-5];hi=np.r_[[-.18,.03,.20],m.jnt_range[:16,1]-1e-5]
q0=np.tile([.05,.3,.3,.15],4)
def fun(v):return np.r_[(tips(v[3:],v[:3])-centers).ravel()*1000,(v[3:]-q0)*.15,(v[:3]-[-.25,-.02,.15])*5]
f=least_squares(fun,np.r_[[-.25,-.02,.15],q0],bounds=(lo,hi),max_nfev=1500)
base=f.x[3:];pos=f.x[:3];press=[]
for k in range(4):
 target=centers[k].copy();target[2]=.095
 def pressfun(v):
  q=base.copy();q[k*4:k*4+4]=v
  return np.r_[(tips(q,pos)[k]-target)*1000,(v-base[k*4:k*4+4])*.1]
 ff=least_squares(pressfun,base[k*4:k*4+4],bounds=(lo[3+k*4:7+k*4],hi[3+k*4:7+k*4]),max_nfev=500)
 q=base.copy();q[k*4:k*4+4]=ff.x;press.append(q.tolist())
r=dict(position=pos.tolist(),orientation=rot.as_quat().tolist(),base=base.tolist(),press=press,tip_local=points.tolist(),rest_tips=tips(base,pos).tolist(),fit_error_mm=((tips(base,pos)-centers)*1000).tolist(),note='CPU FK planning only, no physical validation; right hand')
(out/'coordinator_plan.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
