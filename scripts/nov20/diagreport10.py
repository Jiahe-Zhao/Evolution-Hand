
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from bead10 import *
from scipy.spatial.transform import Slerp
tag='wall4_0';x=np.load(out/(tag+'.npy'));aligned=np.load(out/'aligned.npy');skew=np.load(out/'skew35.npy');c=json.loads((out/'skew35_contacts.json').read_text())
# Verify the interpolated replay in addition to actual Newton states. Far from
# rod, bounding sphere separates outer bead from finite cylinder unequivocally.
mins=[];times=np.arange(len(x));interp=Slerp(times,R.from_quat(x[:,3:7]));ts=np.linspace(0,len(x)-1,(len(x)-1)*8+1);qs=interp(ts).as_quat();ps=np.stack([np.interp(ts,times,x[:,i]) for i in range(3)],axis=1)
for pos,q in zip(ps,qs):
 if np.linalg.norm(pos[:2]-xy)>np.hypot(.010,H)+rad:continue
 pose=np.r_[pos,q];rot=R.from_quat(q).inv();d=rot.apply([0,0,1]);tip=rot.apply(np.r_[xy,zt]-pos);foot=tip-(zt-zb)*d;extent=rad*np.linalg.norm(d[:2])
 if max(tip[2],foot[2])+extent < -H or min(tip[2],foot[2])-extent>H:continue
 depth,gap,angle=measure(pose);mins.append(gap)
min_gap=min(mins);held=x[:,10]>0;idx=np.flatnonzero(held&(x[:,7]>=10)&(x[:,8]>0)&(times>525))[0];lift=(x[held,2].max()-up[2])*1000;skdrift=1000*np.linalg.norm(skew[-120:,:3]-skew[-120,:3],axis=1).max();print(min_gap,lift,x[:,12].max(),x[-1,10],skdrift,skew[-120:,13].mean(),skew[-120:,14].max(),skew[-120:,7].min(),skew[-120:,7].max())
