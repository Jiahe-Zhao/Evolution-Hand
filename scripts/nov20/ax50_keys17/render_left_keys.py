"""Render recorded AX50 poses without modifying physical trajectories."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys
from pathlib import Path
import numpy as np
ROOT=Path((str(_EVOLUTION_ROOT)));pkg=ROOT/'outputs/ax50_visualskin_v3_0_0'
sys.path[:0]=[str(pkg),str(pkg/'tools')]
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
from render_core import encode,strip
from scipy.spatial.transform import Rotation as R
z=np.load(sys.argv[1]);out=Path(sys.argv[2]);poses=z['body_q'];times=z['time'];travel=z['key_travel'];names=list(z['body_names'])
skin=VisualSkin('left',names,device='cpu')
faces=np.array([[0,2,1],[1,2,3],[4,5,6],[5,7,6],[0,1,4],[1,5,4],[2,6,3],[3,6,7],[0,4,2],[2,4,6],[1,3,5],[3,7,5]])
def box(center,half,color,label):
 v=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*half+center
 return v,faces,color,label
centers=np.array([[-.123,.001,.096],[-.128,-.018,.096],[-.118,-.037,.096],[-.097,-.056,.096]])
def camera(eye):
 aim=np.array([-.07,-.025,.10]);f=np.array(eye)-aim;f/=np.linalg.norm(f);x=np.cross([0,0,1],f);x/=np.linalg.norm(x);y=np.cross(f,x);b=np.array([x,-y,f]).T;scale=2200;off=np.array([300,224,0])-aim@b*scale;return b,scale,off
cams=[camera([-.23,-.30,.36]),camera([-.2,.10,.38])]
def frames():
 for i in np.unique(np.searchsorted(times,np.arange(times[0],times[-1],1/30)).clip(0,len(times)-1)):
  v,f=skin.update(poses[i]);parts=[(v,f,(222,174,143),'AX50'),box([-.114,-.028,.080],[.035,.043,.002],[70,85,100],'deck')]
  for k,c in enumerate(centers):
   col=[90,180,110] if travel[i,k]>=.002 else [160,190,220];parts.append(box(c+[0,0,-travel[i,k]],[.009,.0075,.002],col,f'key{k+1}'))
  images=[panel(parts,c) for c in cams]
  text=' / '.join(f'K{k+1} {travel[i,k]*1000:.2f} mm' for k in range(4))
  yield strip(images,['Oblique','Contact view'],[[f't = {times[i]:.3f} s',text],['Recorded physics; display skin', 'Trigger 2 mm / release 0.5 mm']],'AX50 scene 17 - recorded key operation')
encode(out,float(sys.argv[3]) if len(sys.argv)>3 else 30,frames())
