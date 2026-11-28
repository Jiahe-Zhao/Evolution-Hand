
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
def checkpoint_data(tag):
 base=np.load(OUT/'hand_v6.npz');c=np.load(OUT/(tag+'_checkpoint_239.npz'));n=base['particles'].shape[1]
 return {'body_q':c['body_q'][None], 'time':np.array([float(c['time'])]), 'body_names':base['body_names'], 'tri':base['tri'], 'particles':c['particle_q'][None,:n], 'leaf_q':c['joint_q'][-2:][None]}
from pathlib import Path
import sys,json,csv
import numpy as np,mujoco as mj,trimesh
ROOT=Path((str(_EVOLUTION_ROOT)));pkg=ROOT/'outputs/ax50_visualskin_v3_0_0';sys.path[:0]=[str(pkg),str(pkg/'tools')]
import ax50_visualskin,warp as wp
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
from render_core import encode,strip
from scipy.spatial.transform import Rotation as R
OUT=ROOT/'outputs/nov20/ax50_leaf04';tag=sys.argv[1];z=checkpoint_data(tag);poses=z['body_q'];times=z['time'];skin=VisualSkin('left',list(z['body_names']),device='cpu');m=mj.MjModel.from_xml_path(str(OUT/'leaf_clean.xml'));d=mj.MjData(m)
soft_tri=z['tri'];soft_tri=soft_tri[(soft_tri>=0).all(axis=1)&(soft_tri<z['particles'].shape[1]).all(axis=1)]
parts=[]
for g in range(m.ngeom):
 name=m.geom(g).name
 if name=='floor':continue
 typ=m.geom_type[g];sz=m.geom_size[g]
 if typ==mj.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(extents=sz*2);v=mesh.vertices;f=mesh.faces
 elif typ==mj.mjtGeom.mjGEOM_CAPSULE:
  mesh=trimesh.creation.capsule(radius=sz[0],height=sz[1]*2,count=[8,12]);v=np.array(mesh.vertices);v-=mesh.bounds.mean(0);f=mesh.faces
 elif typ==mj.mjtGeom.mjGEOM_MESH:
  k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]]
 else:continue
 parts.append((g,np.array(v),np.array(f),np.array(m.geom_rgba[g,:3])*255,name))
def scene(i):
 d.qpos[:]=z['leaf_q'][i];mj.mj_kinematics(m,d);v,f=skin.update(poses[i]);p=[(v,f,[222,174,143],'AX50 display skin'),(z['particles'][i],soft_tri,[235,187,156],'AX50 simulated soft surfaces')]
 for g,v,f,col,name in parts:p.append((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g],f,col,name))
 return p
@wp.kernel
def visibility(mesh:wp.uint64,eye:wp.vec3,ends:wp.array[wp.vec3],visible:wp.array[int]):
 i=wp.tid();dv=ends[i]-eye;dist=wp.length(dv);q=wp.mesh_query_ray(mesh,eye,dv/dist,dist-1.e-5);visible[i]=wp.int32(not q.result)
eye=np.array([-.121,.009,.35]);gx,gy=np.meshgrid(((np.arange(40)+.5)/40*2-1)*.011,((np.arange(56)+.5)/56*2-1)*.016);ends=np.c_[gx.ravel()-.121,gy.ravel()+.009,np.full(2240,.036)];endpoint=wp.array(ends,dtype=wp.vec3,device='cuda:0');visible=wp.zeros(2240,dtype=int,device='cuda:0');raymesh=None;ratios=[]
def pack(p):
 vv=[];ff=[];offset=0
 for v,f,col,name in p:
  if name=='food':continue
  vv.append(v);ff.append(f+offset);offset+=len(v)
 return np.concatenate(vv).astype(np.float32),np.concatenate(ff).astype(np.int32).ravel()
def camera(aim,view,scale):
 zz=np.asarray(view,dtype=float);zz/=np.linalg.norm(zz);xx=np.cross([0,0,1],zz);xx/=np.linalg.norm(xx);b=np.array([xx,-np.cross(zz,xx),zz]).T;return b,scale,np.array([300,224,0])-np.asarray(aim)@b*scale
side=camera([-.07,.005,.085],[-.4,-1,.65],2350)
def auditparts(p):
 out=[];focal=224/np.tan(np.deg2rad(19/2))
 for v,f,col,name in p:
  depth=eye[2]-v[:,2];w=np.c_[focal*(v[:,0]-eye[0])/depth,-focal*(v[:,1]-eye[1])/depth,v[:,2]];out.append((w,f,col,name))
 return out
from PIL import Image
p=scene(0);v,f=pack(p);raymesh=wp.Mesh(points=wp.array(v,dtype=wp.vec3,device='cuda:0'),indices=wp.array(f,dtype=int,device='cuda:0'));wp.launch(visibility,2240,inputs=[raymesh.id,wp.vec3(*eye),endpoint,visible],device='cuda:0');ratio=float(visible.numpy().mean());frame=strip([panel(p,side),panel(auditparts(p),(np.eye(3),1,np.array([300,224,0])))],['Actual recorded AX50 state','Original fixed audit camera'],[[f't={times[0]:.2f} s'],[f'Visible: {ratio*100:.1f}%']],'AX50 leaf 04 - physical checkpoint preview');frame.save(OUT/(tag+'_checkpoint.png'));print('CHECKPOINT_VISIBILITY',ratio)
