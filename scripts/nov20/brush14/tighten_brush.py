
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
from pathlib import Path
from xml.etree import ElementTree as E
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));s=np.load(P/'pose.npz');q=s['q'].copy();center=s['center'];r=_evo_parse_xml(P/'pose.xml').getroot();ob=r.find("worldbody/body[@name='brush']");E.SubElement(ob,'geom',name='axis',type='capsule',fromto='0 0 -.035 0 0 .095',size='.00005',mass='.0000001',contype='0',conaffinity='0');m=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));d=mj.MjData(m);pads=[m.geom('handgeom'+str(i)).id for i in [20,40,60,80]];rod=m.geom('shaft').id;axis=m.geom('axis').id;rad=np.array([[0,1,0],[0,-1,0],[-.75,-.66,0],[-.9,.43,0]]);heights=[.005,.005,-.008,-.020];other=[i for i in range(m.ngeom) if i not in pads+[rod,axis]]
def fun(x,depth):
 d.qpos[:27]=q;d.qpos[6:23]=x;mj.mj_kinematics(m,d);z=np.zeros(6);res=[]
 for j,i in enumerate(pads):
  mj.mj_geomDistance(m,d,i,axis,.25,z);res.extend(1000*(z[:3]-(center+rad[j]*(.005-depth)+[0,0,heights[j]])))
 for i in other:res.append(1.5*min(0,1000*(mj.mj_geomDistance(m,d,i,rod,.008,z)-.0003)))
 return np.r_[res,.06*(x-q[6:23])]
lo=m.jnt_range[6:23,0]+1e-5;hi=m.jnt_range[6:23,1]-1e-5;v=q[6:23].copy()
for depth,key in [(.00015,'initial'),(.001,'target')]:
 sol=least_squares(fun,np.clip(v,lo,hi),args=(depth,),bounds=(lo,hi),diff_step=1e-4,max_nfev=160);v=sol.x;print(key,np.linalg.norm(fun(v,depth)),fun(v,depth)[:12]);out=q.copy();out[6:23]=v
 if key=='initial':initial=out
 else:control=out
np.savez(P/'grasp_seed.npz',q=initial,control=control,center=center)
