
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import trimesh
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush';PKG=ROOT/'outputs/ax50_visualskin_v3_0_0';sys.path[:0]=[str(PKG),str(PKG/'tools')]
from ax50_visualskin import VisualSkin
from render_core import basis,strip,encode
from smooth_render import panel
D=np.load(P/'grasp_trial.npz');poses=D['body_q'];names=list(D['body_names']);meta=json.loads((P/'grasp_trial.json').read_text());skin=VisualSkin('left',names);rb=names.index('brush');root=np.array(poses[0,rb,:3]);paperz=meta['paper_top_z'];cyl=trimesh.creation.cylinder(radius=.005,height=.130,sections=40);cyl.vertices[:,2]+=.030
paper=trimesh.creation.box(extents=[.18,.12,.002]);paper.vertices+=root+[.025,0,paperz-root[2]-.001]
wood=trimesh.creation.box(extents=[.20,.14,.006]);wood.vertices+=root+[.025,0,paperz-root[2]-.005]
ids=np.array([[names.index(f'bristle_{i:02d}_{j:02d}_y') for j in range(5)] for i in range(13)])
# Same 900 strand display count, driven by the recorded physical bundle poses.
rr=np.array([[0.,0.]]+[[.0035*(.5 if b<=6 else 1)*np.cos(2*np.pi*((b-1)%6)/6+(0 if b<=6 else np.pi/6)),.0035*(.5 if b<=6 else 1)*np.sin(2*np.pi*((b-1)%6)/6+(0 if b<=6 else np.pi/6))] for b in range(1,13)])
tri=Delaunay(rr);rng=np.random.default_rng(14019);samples=[]
while len(samples)<900:
 xy=rng.uniform(-.0035,.0035,2)
 if tri.find_simplex(xy)>=0:samples.append(xy)
samples=np.array(samples);si=tri.find_simplex(samples);bc=np.einsum('nij,nj->ni',tri.transform[si,:2],samples-tri.transform[si,2]);weights=np.c_[bc,1-bc.sum(1)];verts=tri.simplices[si];length=1-rng.uniform(0,.02,900)-.12*(np.linalg.norm(samples,axis=1)/.0035)**2;length[::12]=1
angles=np.arange(5)*2*np.pi/5;unit=np.c_[np.cos(angles),np.sin(angles)];faces=[]
for h in range(900):
 for j in range(5):
  for k in range(5):
   a=h*30+j*5+k;b=h*30+j*5+(k+1)%5;faces.extend([[a,b,a+5],[b,b+5,a+5]])
faces=np.array(faces)
def hairs(q):
 guides=[]
 for group in ids:
  line=[]
  for j,b in enumerate(group):
   pos=q[b,:3];rot=R.from_quat(q[b,3:]);line.append(pos)
   if j==4:line.append(pos+rot.apply([0,0,-.0044]))
  guides.append(line)
 guides=np.array(guides);nodes=np.einsum('nk,nkjc->njc',weights,guides[verts]);u=length[:,None]*np.arange(6)[None,:];ix=np.minimum(u.astype(int),4);frac=(u-ix)[...,None];points=nodes[np.arange(900)[:,None],ix]*(1-frac)+nodes[np.arange(900)[:,None],ix+1]*frac
 tan=np.gradient(points,axis=1);tan/=np.maximum(np.linalg.norm(tan,axis=2,keepdims=True),1e-12);ref=np.broadcast_to([1.,0,0],tan.shape);normal=ref-(ref*tan).sum(2,keepdims=True)*tan;normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-12);binormal=np.cross(tan,normal);radius=.000032*(1-.72*(np.arange(6)/5)**2)
 v=points[:,:,None,:]+radius[None,:,None,None]*(normal[:,:,None,:]*unit[None,None,:,0,None]+binormal[:,:,None,:]*unit[None,None,:,1,None]);return v.reshape(-1,3),faces,[103,71,38],'900_display_hairs'
def camera(center,view,scale):
 b=basis(view,(0,0,1));return b,scale,np.array([300,224,0])-np.asarray(center)@b*scale
v,f=skin.update(poses[0]);allv=np.vstack([v,paper.vertices,cyl.vertices+root]);cam=camera((allv.min(0)+allv.max(0))/2,(.5,1,.5),2100)
macro=camera(root+[0,0,-.045],(.6,1,.2),9500)
b=basis((0,0,1),(0,1,0));top=(b,3500,np.array([300,224,0])-(root+[.025,0,paperz-root[2]])@b*3500)
paths=json.loads((P/'tongji_paths.json').read_text());target=[]
# Target guide is blue and explicitly marked; no fabricated contact ink.
for stroke in paths['strokes']:
 xy=np.array(stroke['xy_m'])+root[:2]
 for a,bp in zip(xy[:-1],xy[1:]):
  dv=bp-a;normal=np.array([-dv[1],dv[0]])/max(np.linalg.norm(dv),1e-12)*.00015;vs=np.array([np.r_[a+normal,paperz+.00005],np.r_[a-normal,paperz+.00005],np.r_[bp+normal,paperz+.00005],np.r_[bp-normal,paperz+.00005]]);target.append((vs,np.array([[0,1,2],[1,3,2]]),[65,128,190],'target_not_ink'))
def frame(k):
 q=poses[k];v,f=skin.update(q);shaft=(R.from_quat(q[rb,3:]).apply(cyl.vertices)+q[rb,:3],cyl.faces,[145,91,43],'shaft');hair=hairs(q);props=[(wood.vertices,wood.faces,[138,110,75],'table'),(paper.vertices,paper.faces,[250,249,242],'paper')]+target
 a=panel(props+[shaft,hair,(v,f,[217,161,130],'AX50 display skin')],cam);bb=panel(props+[shaft,hair],macro);cc=panel(props+[hair],top)
 return strip([a,bb,cc],['AX50 VisualSkin v3.0.0','Brush detail / hand hidden','Paper target / hand + shaft hidden'],[['Shared Newton scene','Diagnostic grasp attempt'],['900 visual hairs / 13 guides','No fabricated ink'],['Blue = Tongji target, NOT actual ink',f'Time {k*.01:.2f} s']],'AX50 + brush | integration diagnostic, NOT accepted writing')
for k in [0,len(poses)-1]:frame(k).save(P/f'ax50_brush_{k:03d}.png')
encode(P/'ax50_brush_diagnostic.mp4',25,(frame(k) for k in range(len(poses))))
print('rendered',len(poses),flush=True)
