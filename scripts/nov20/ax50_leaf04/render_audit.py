
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys,json,csv
import numpy as np,mujoco as mj,trimesh
ROOT=Path((str(_EVOLUTION_ROOT)));pkg=ROOT/'outputs/ax50_visualskin_v3_0_0';sys.path[:0]=[str(pkg),str(pkg/'tools')]
import ax50_visualskin,warp as wp
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
from render_core import encode,strip
from scipy.spatial.transform import Rotation as R
OUT=ROOT/'outputs/nov20/ax50_leaf04';tag=sys.argv[1];z=np.load(OUT/(tag+'.npz'));poses=z['body_q'];times=z['time'];skin=VisualSkin('left',list(z['body_names']),device='cpu');m=mj.MjModel.from_xml_path(str(OUT/'leaf_clean.xml'));d=mj.MjData(m)
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
for i in range(len(times)):
 if not np.isfinite(poses[i]).all():ratios.append(float('nan'));continue
 v,f=pack(scene(i))
 if raymesh is None:raymesh=wp.Mesh(points=wp.array(v,dtype=wp.vec3,device='cuda:0'),indices=wp.array(f,dtype=int,device='cuda:0'))
 else:raymesh.points.assign(v);raymesh.refit()
 wp.launch(visibility,2240,inputs=[raymesh.id,wp.vec3(*eye),endpoint,visible],device='cuda:0');ratios.append(float(visible.numpy().mean()))
 if i%50==0:print('VIS',i,ratios[-1],flush=True)
np.savetxt(OUT/(tag+'_visibility.csv'),np.c_[times,ratios],delimiter=',',header='time_s,visible_fraction',comments='')
def camera(aim,view,scale):
 zz=np.asarray(view,dtype=float);zz/=np.linalg.norm(zz);xx=np.cross([0,0,1],zz);xx/=np.linalg.norm(xx);b=np.array([xx,-np.cross(zz,xx),zz]).T;return b,scale,np.array([300,224,0])-np.asarray(aim)@b*scale
side=camera([-.07,.005,.085],[-.4,-1,.65],2350)
def auditparts(p):
 out=[];focal=224/np.tan(np.deg2rad(19/2))
 for v,f,col,name in p:
  depth=eye[2]-v[:,2];w=np.c_[focal*(v[:,0]-eye[0])/depth,-focal*(v[:,1]-eye[1])/depth,v[:,2]];out.append((w,f,col,name))
 return out
trace=list(csv.DictReader((OUT/(tag+'_trace.csv')).open()));valid=np.array([(r>=.8 and float(t['leaf0_index_N'])>.01 and float(t['leaf1_middle_N'])>.01 and sum(float(t[k]) for k in ['leaf0_middle_N','leaf0_other_N','leaf1_index_N','leaf1_other_N'])==0) for r,t in zip(ratios,trace)])
has_substep_audit='qualified_substeps' in trace[0]
if has_substep_audit:valid &= np.array([int(t['qualified_substeps'])==int(t['substeps']) and int(t['substeps'])>0 for t in trace])
idx=np.flatnonzero(valid);groups=np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else [];best=max(groups,key=len) if groups else [];duration=float(times[best[-1]]-times[best[0]]) if len(best) else 0.;result=dict(tag=tag,initial_visibility=ratios[0],max_visibility=float(np.nanmax(ratios)),end_visibility=ratios[-1],qualified_hold_s=duration,passed=bool(duration>=.5 and np.isfinite(poses).all()),scope='fixed initial condition; actual rigid poses; opaque AX50 display skin and simulated soft surfaces included in 2240 fixed-camera rays',hold_interval_s=[float(times[best[0]]),float(times[best[-1]])] if len(best) else [])
status=json.loads((OUT/(tag+'_status.json')).read_text())
result['simulation_status']=status['stop']
if has_substep_audit:
 result['audit_endpoint_max_error_N']=max(float(t['audit_end_load_error_N']) for t in trace)
 result['passed']=bool(result['passed'] and result['audit_endpoint_max_error_N']<.0006)
result['contact_audit']='all physics substeps; window min/max saved' if has_substep_audit else '100 Hz endpoint samples only'
if has_substep_audit and len(best):
 result['qualified_interval_min_tip_N']=[min(float(trace[i][key]) for i in best) for key in ['min_leaf0_index_N','min_leaf1_middle_N']]
 result['qualified_interval_max_unassigned_N']=max(float(trace[i]['max_unassigned_N']) for i in best)
result['final_leaf_degrees']=np.rad2deg(z['leaf_q'][-1]).tolist()
result['returned_below_1_degree']=bool(np.max(np.abs(np.rad2deg(z['leaf_q'][-1])))<1.)
result['passed']=bool(result['passed'] and status['stop']=='completed' and all(t['finite']=='True' for t in trace) and result['returned_below_1_degree'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
axes[0].plot(times,np.array(ratios)*100);axes[0].axhline(80,color='r',linestyle='--');axes[0].set_ylabel('Target visible (%)')
axes[1].plot(times,np.rad2deg(z['leaf_q']));axes[1].set_ylabel('Leaf angle (deg)');axes[1].legend(['Leaf 0','Leaf 1'])
for key in ['leaf0_index_N','leaf1_middle_N','leaf0_other_N','leaf1_other_N']:
 axes[2].plot(times,[float(t[key]) for t in trace],label=key)
if has_substep_audit:
 for a,b,col,label in [('min_leaf0_index_N','max_leaf0_index_N','C0','index substep range'),('min_leaf1_middle_N','max_leaf1_middle_N','C1','middle substep range')]:
  axes[2].fill_between(times,[float(t[a]) for t in trace],[float(t[b]) for t in trace],color=col,alpha=.2,label=label)
axes[2].set_ylabel('Normal contact (N)');axes[2].set_xlabel('Simulation time (s)');axes[2].legend()
for ax in axes:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/(tag+'_curves.png'),dpi=160);plt.close(fig)
(OUT/(tag+'_result.json')).write_text(json.dumps(result,indent=2));print(result,flush=True)
def frames():
 for i in np.unique(np.searchsorted(times,np.arange(times[0],times[-1],1/30)).clip(0,len(times)-1)):
  if not np.isfinite(poses[i]).all():continue
  p=scene(i);yield strip([panel(p,side),panel(auditparts(p),(np.eye(3),1,np.array([300,224,0])))],['AX50 physical operation','Fixed audit camera; hand included'],[[f't={times[i]:.2f}s  |  playback 0.5x','Leaf angles '+str(np.rad2deg(z['leaf_q'][i]).round(1))],[f'Target visible {ratios[i]*100:.1f}%','Acceptance: >=80% for 0.5s with both tips supporting']],'AX50 04 - independently hinged leaves')
encode(OUT/(tag+'.mp4'),15,frames())
