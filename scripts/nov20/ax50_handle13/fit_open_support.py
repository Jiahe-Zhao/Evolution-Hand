from pathlib import Path
import numpy as np,mujoco as mj,json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_handle13';m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m)
conv=json.loads((ROOT/'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json').read_text())['canonical_to_native'];sign=np.array([conv[m.joint(i).name.split('_Joints_')[-1]] for i in range(16)])
ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith('_'+n)) for n in ['link_3_2','link_4_2']];points=np.array([[0,.0044,.012],[0,.0044,.012]]);goals=np.array([[-.112,-.035,.047],[-.112,-.010,.047]]);rot0=R.from_euler('x',180,degrees=True)*R.from_quat([0,.707106781,-.707106781,0])
def evaluate(x):
 q=np.zeros(16);q[4]=x[3];q[8]=x[4];d.qpos[:]=0;d.qpos[:16]=q*sign;mj.mj_kinematics(m,d);rot=R.from_euler('z',x[5])*rot0
 tip=np.array([rot.apply(d.xpos[i]+d.xmat[i].reshape(3,3)@v-[0,0,.3])+x[:3] for i,v in zip(ids,points)]);return q,rot,tip
def residual(x):
 q,rot,tip=evaluate(x);return np.r_[((tip-goals)*1000).ravel(),x[3:5]*4]
f=least_squares(residual,[.0,-.01,.042,0,0,-.35],bounds=([-.05,-.1,.02,-.15,-.15,-.65],[.08,.05,.07,.5,.5,.65]));q,rot,tip=evaluate(f.x);geom={}
for name in ['link_3_3','link_4_3']:
 b=next(i for i in range(m.nbody) if m.body(i).name.endswith('_'+name));g=next(i for i in range(m.ngeom) if m.geom_bodyid[i]==b and '/skin_' in m.geom(i).name);k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];rg=R.from_quat(m.geom_quat[g][[1,2,3,0]]);v=rg.apply(v)+m.geom_pos[g];v=v@d.xmat[b].reshape(3,3).T+d.xpos[b]-[0,0,.3];v=rot.apply(v)+f.x[:3];geom[name]=dict(min_world_m=v.min(0).tolist(),max_world_m=v.max(0).tolist(),x_clearance_to_vessel_front_m=float(v[:,0].min()+.160))
r=dict(position=f.x[:3].tolist(),orientation=rot.as_quat().tolist(),open_pose=q.tolist(),contact_bodies=['link_3_2','link_4_2'],palmar_local_points=points.tolist(),target_points=goals.tolist(),actual_points=tip.tolist(),error_mm=((tip-goals)*1000).tolist(),distal_geometry=geom,scope='CPU FK candidate only; all MCP/PIP/DIP zero before support, arch positioning only; no physical pass')
(OUT/'open_support_plan.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
