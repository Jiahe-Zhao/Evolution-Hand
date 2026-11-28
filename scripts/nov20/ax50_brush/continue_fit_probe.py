
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys, json, os
import numpy as np
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush';OUT=P/'continue'
sys.path.insert(0,str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
import ax50_visualskin
sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
from continue_opposed_trial import prepare,Q,MOUNT,R
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution
r=AX50WritingScene(side='left',wrist='locked',position=MOUNT,orientation=Q,gravity=(0,0,0),iterations=60,brush_xml=P/'tongji_scene_source.xml')
r.restore_prepared_hand(np.load(prepare()))
q0=r.state.joint_q.numpy()
pids=np.flatnonzero(np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]=='link_2_3')+r.pad_offset
a_bone=r.body_names_all.index('link_2_3')
rest=r.state.body_q.numpy()[a_bone]
pad_local=(r.state.particle_q.numpy()[pids]-rest[:3])@Rotation.from_quat(rest[3:]).as_matrix()
shape=r.model.shape_label.index('/ax50_left/Geometry/link_1_2/skin_1_2')
mesh=r.model.shape_source[shape]
thumb_vertices=np.asarray(mesh.vertices,dtype=float)
thumb_transform=np.asarray(r.model.shape_transform.numpy()[shape])
thumb_rotation=Rotation.from_quat(thumb_transform[3:])

def transform(p,point):
 return Rotation.from_quat(p[3:]).apply(point)+p[:3]
def evaluate(v,verbose=False):
 q=q0.copy()
 for n,x in zip(['digit_2_flex_1','digit_2_flex_2','thumb_sweep','digit_1_flex_2'],v):q[r.joint_names.index(n)]=x
 r.state.joint_q.assign(q)
 import newton
 newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state)
 body=r.state.body_q.numpy()
 index=transform(body[a_bone],pad_local)
 bp=body[r.model.shape_body.numpy()[shape]]
 thumb=Rotation.from_quat(bp[3:]).apply(thumb_rotation.apply(thumb_vertices)+thumb_transform[:3])+bp[:3]
 index=R.inv().apply(index-MOUNT)
 thumb=R.inv().apply(thumb-MOUNT)
 if verbose:print('index',index.min(0),index.max(0),'thumb',thumb.min(0),thumb.max(0),flush=True)
 results=[]
 for y in np.arange(.040,.064,.002):
  sel=(thumb[:,1]>y-.005)&(thumb[:,1]<y+.015)
  if verbose:print('SLICE',y,int(sel.sum()),int((np.abs(index[:,1]-y)<.02).sum()),flush=True)
  for x in np.arange(.024,.064,.002):
   for z in np.arange(-.020,.016,.002):
    # XZ plane: localY is aligned with mounted pen axis.
    axis=np.array([x,0,z]);tv=thumb[sel];iv=index[np.abs(index[:,1]-y)<.02]
    if not len(tv) or not len(iv):continue
    tr=np.linalg.norm(((tv[:,[0,1,2]]-axis)*np.array([1,0,1])),axis=1)
    ir=np.linalg.norm(((iv[:,[0,1,2]]-axis)*np.array([1,0,1])),axis=1)-.005
    tg=tv[np.argmin(tr)];ig=iv[np.argmin(ir)]
    gap=0.0
    t_du=float(tr.min()*.001) if tr.min()>=5 else (tr.min()-5)
    tgap=(np.linalg.norm(tg-axis)*1000-5)
    score=abs(ir.min())+abs(tgap)+100*max(0,tgap-2)+100*max(0,-2-tgap)+10*gap
    results.append((score,y,x,z,float(tr.min()-5),float(ir.min()),tg.tolist(),ig.tolist()))
 if not results:return (1e3,0,0,0,0,0,[],[])
 best=min(results)
 if verbose:print('BEST',best,flush=True)
 return best
if os.getenv('OPTIMIZE'):
 def objective(v):
  b=evaluate(v)
  return b[0]
 res=differential_evolution(objective,[(.7,1.5),(.3,1.3),(.5,.95),(.6,1.5)],popsize=5,maxiter=8,polish=True,workers=1)
 print('OPT',res.x,res.fun,flush=True)
 for _ in range(2):evaluate(res.x,verbose=True)
else:
 for k,v in enumerate([[1.5,.60484,.95,1.5],[1.5,.9,.95,.8],[1.5,1.0,.75,.7]]):evaluate(v,verbose=True)
