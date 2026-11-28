
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from scipy.optimize import least_squares
def fit(r,q):
 from make_long import np,mj,E,axis,center,loc,fmt,os
 import copy
 r=copy.deepcopy(r);E.SubElement(r.find("worldbody/body[@name='pen']"),'geom',name='ikaxis',type='capsule',fromto=fmt(np.r_[loc(-.022),loc(.105)]),size='.00015',mass='.0000001')
 mod=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));data=mj.MjData(mod);data.qpos[:27]=q[:27];data.qpos[27:]=q[[27,28,29,33,30,31,32]];orig=q[6:19].copy();orig[-4:]=orig[5:9];ids=[mod.geom('handgeom'+str(j)).id for j in range(101) if j not in (20,40,60)];rod=[mod.geom(k).id for k in ('front','rear')];tips=[mod.geom('handgeom'+str(j)).id for j in (20,40,60)]
 normal=data.geom_xpos[tips[0]]-data.geom_xpos[tips[1]] if False else np.array([0.,1.,-.08]);normal-=normal.dot(axis)*axis;normal/=np.linalg.norm(normal)
 def loss(x,depth):
  data.qpos[6:19]=x;mj.mj_kinematics(mod,data);z=np.zeros(6)
  dist=lambda i:min(mj.mj_geomDistance(mod,data,i,j,.015,z) for j in rod)*1000
  out=[]
  for j,(i,sgn) in enumerate(zip(tips,[1,-1,-1])):
   zj=np.zeros(6);mj.mj_geomDistance(mod,data,i,mod.geom('ikaxis').id,.08,zj)
   rad=sgn*normal if j!=2 else -.65*normal+.76*np.cross(axis,normal)
   out.extend(1000*(zj[:3]-center)-(float(os.getenv('SP','12'))*axis if j==2 else 0)-(4.5-depth)*rad)
  return np.r_[out, [min(0,dist(i)-.15) for i in ids], .12*(x-orig)]
 lim=mod.jnt_range[6:19];x=np.clip(orig,lim[:,0]+.0001,lim[:,1]-.0001)
 for depth in [.08,float(os.getenv('PINCH','1.6'))]:
  res=least_squares(loss,x,args=(depth,),bounds=(lim[:,0],lim[:,1]),diff_step=.0001,max_nfev=120);x=res.x
  print('fit',depth,loss(x,depth)[:2],np.linalg.norm(loss(x,depth)),flush=True)
  if depth==.08:q[6:19]=x
 return x
