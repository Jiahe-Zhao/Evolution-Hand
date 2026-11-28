
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
def setup(path):
 m=mj.MjModel.from_xml_path(str(path));d=mj.MjData(m)
 keys=['link_1_thumb_spread_joint','link_0_0_to_link_1_0','link_1_0_to_link_1_1','link_1_1_to_link_1_2','link_1_2_to_link_1_3']
 js=[m.joint(k).id for k in keys];idx=m.jnt_qposadr[js];bounds=m.jnt_range[js].T
 body=m.body('link_1_3').id;g=np.where(m.geom_bodyid==body)[0][0]
 def fk(q):
    d.qpos[idx]=q;mj.mj_kinematics(m,d);return d.geom_xpos[g].copy()
 def solve(x,start):
    r=least_squares(lambda q:np.r_[100*(fk(q)-x),.01*(q-start)],start,bounds=bounds,max_nfev=120)
    return r.x,fk(r.x)
 if __name__=='__main__':
    rng=np.random.default_rng(0);pts=np.array([fk(rng.uniform(*bounds)) for i in range(5000)]);print('range',pts.min(0),pts.max(0));print(pts[(pts[:,2]>.135)&(pts[:,2]<.145)][::12][:45])
 return keys,solve
if __name__=='__main__':
 from build import OUT
 setup(OUT/'phone.xml')
