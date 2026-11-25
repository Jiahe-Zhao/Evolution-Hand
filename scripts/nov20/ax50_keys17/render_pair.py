import sys,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/zjh/Evolution_PC');pkg=ROOT/'outputs/ax50_visualskin_v3_0_0';sys.path[:0]=[str(pkg),str(pkg/'tools')]
from ax50_visualskin.visual import VisualSkin
from render_core import CollisionSurfaces,encode,strip
from smooth_render import panel
z=np.load(sys.argv[1]);out=Path(sys.argv[2]);label=sys.argv[3];names=list(z['body_names']);skin=VisualSkin('left',names,device='cpu');coll=CollisionSurfaces(pkg/'ax50_visualskin/assets/left',names);coll.highlight={'skin_5_1':[240,100,40]}
poses=z['body_q'];p=z['particles'];times=z['time'];tri=z['tri'];tri=tri[np.max(tri,axis=1)<p.shape[1]]
# Whole hand, palm-facing, plus magnified true contact surfaces.
allpts=p[0];aim=(allpts.min(0)+allpts.max(0))/2
bi=names.index('link_5_1');focus=poses[0,bi,:3]+[0,0,0]
def camera(aim,view,scale):
 f=np.asarray(view,dtype=float);f/=np.linalg.norm(f);x=np.cross([0,0,1],f);x/=np.linalg.norm(x);y=np.cross(f,x);b=np.array([x,-y,f]).T;return b,scale,np.array([300,224,0])-np.asarray(aim)@b*scale
c1=camera(aim,[-.3,-1,1],2200);c2=camera(focus,[-.3,-1,1],6500)
def frames():
 for i in range(len(times)):
  v,f=skin.update(poses[i]);physical=coll.pose(poses[i],['skin_5_1','skin_5_2','skin_5_3','link_5_1'])+[(p[i],tri,[170,90,90],'soft tissue')]
  yield strip([panel([(v,f,[222,174,143],'AX50')],c1),panel(physical,c2)],['Whole hand - display skin','Actual soft mesh / little finger collider'],[[f'physics t={times[i]:.2f}s','Playback 4x slow'],['Orange: proximal cover','Red: physical soft tissue']],label)
encode(out,25,frames())
