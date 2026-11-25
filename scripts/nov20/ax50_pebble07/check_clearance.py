from pathlib import Path
import numpy as np,mujoco as mj,json
from scipy.spatial.transform import Rotation as R
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_pebble07';c=json.loads((OUT/'pinch_candidate.json').read_text());m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m);rot=R.from_quat(c['orientation']);base=np.array(c['position'])
results=[]
for label,other in [('straight',[0,0,0,0]),('folded',[.1,.65,1.,.8]),('folded2',[.1,1.,1.2,1.])]:
 d.qpos[:]=0;d.qpos[:21]=c['joint_pose']
 for k in [4,8,12]:d.qpos[k:k+4]=other
 mj.mj_kinematics(m,d);rows=[]
 for g in range(m.ngeom):
  b=int(m.geom_bodyid[g]);name=m.body(b).name
  if 'ax50' not in name or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
  typ=m.geom_type[g];sz=m.geom_size[g]
  if typ==mj.mjtGeom.mjGEOM_MESH:
   k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
  elif typ==mj.mjtGeom.mjGEOM_BOX:v=np.array([[x,y,z] for x in [-sz[0],sz[0]] for y in [-sz[1],sz[1]] for z in [-sz[2],sz[2]]])
  else:
   radius=sz[0];half=sz[1] if typ in [mj.mjtGeom.mjGEOM_CAPSULE,mj.mjtGeom.mjGEOM_CYLINDER] else 0
   v=np.array([[x,y,z] for x in [-radius,radius] for y in [-radius,radius] for z in [-half-radius,half+radius]])
  points=rot.apply(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-[0,0,.3])+base
  inside=(points[:,0]>=-.155)&(points[:,0]<=-.035)&(abs(points[:,1])<=.048)
  if inside.any():
   j=np.flatnonzero(inside)[np.argmin(points[inside,2])];rows.append(dict(body=name,geom=m.geom(g).name,min_z_mm=float(points[j,2]*1000),point=points[j].tolist()))
 rows=sorted(rows,key=lambda x:x['min_z_mm']);results.append(dict(label=label,other_pose=other,lowest=rows[:12]))
(OUT/'clearance_candidates.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
