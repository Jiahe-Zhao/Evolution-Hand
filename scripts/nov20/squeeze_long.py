
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.optimize import least_squares
def tighten(r,q,control,depth):
 mod=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));md=mj.MjData(mod);md.qpos[:]=q[np.r_[np.arange(30),33,30,31,32]];rods=[mod.geom(x).id for x in ('front','rear')];tips=[mod.geom('handgeom'+str(i)).id for i in (20,40)];z=np.zeros(6)
 def fun(x):
  md.qpos[6:15]=x;mj.mj_kinematics(mod,md);return np.r_[[1000*min(mj.mj_geomDistance(mod,md,i,j,.05,z) for j in rods)+depth for i in tips],2*(x-q[6:15])]
 lim=mod.jnt_range[6:15];x=np.clip(q[6:15],lim[:,0]+.0001,lim[:,1]-.0001);res=least_squares(fun,x,bounds=(lim[:,0],lim[:,1]),max_nfev=60,diff_step=.0001);control[:9]=res.x
 return control
