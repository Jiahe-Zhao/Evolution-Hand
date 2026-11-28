
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial import cKDTree
def stats(o,tag):
 a=np.load(o/(tag+'.npz'))['a'];start=np.loadtxt(o/(tag+'_origin.txt'));u=np.linspace(0,1,5001);curve=np.c_[start[0]+.046*u,start[1]+.006*np.sin(2*np.pi*u)];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];marked=(a[:,9]>.01)&np.isfinite(a[:,7]);held=(a[:,10:12]>.01).all(axis=1);ok=marked&held;ind=np.flatnonzero(ok);err,idx=cKDTree(curve).query(a[ok,7:9]);s=arc[idx];end=-1;tot=0
 for x,y in sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ind[j+1]-ind[j]==1):tot+=max(0,y-max(x,end));end=max(end,y)
 return 100*tot/arc[-1],(1000*np.sqrt(np.mean(err**2)) if len(err) else np.nan),bool(marked.any() and held[marked].all()),held.mean(),a[-1,4:7],a[:,9].max()
if __name__=='__main__':
 import sys
 for t in sys.argv[1:]:print(t,stats(OUT/'pen140',t))
