
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from concurrent.futures import ThreadPoolExecutor
import os,sys,subprocess
from build import *
from scipy.spatial import cKDTree
o=OUT/'pen14';pos=np.load(o/'origin.npy');u=np.linspace(0,1,4001);curve=np.c_[pos[0]-.002+.046*u,pos[1]+.006*np.sin(2*np.pi*u)];tree=cKDTree(curve);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
def f(k):
 force,soft,gain=k;tag='s'+str(force)+'_'+str(soft)+'_'+str(gain)
 with open('/tmp/pen14'+tag,'w') as log: subprocess.run([sys.executable,'pen14.py','write',tag],env=dict(os.environ,FORCE=str(force),SOFT=f'{soft} 1.4',GAIN=str(gain)),stdout=log,stderr=log,check=True)
 a=np.load(o/(tag+'.npz'))['a'];marked=(a[:,9]>.01)&np.isfinite(a[:,7]);held=(a[:,10:12]>.01).all(axis=1);ok=marked&held
 ind=np.flatnonzero(ok);err,idx=tree.query(a[ok,7:9]);s=arc[idx];ints=sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ind[j+1]-ind[j]==1);end=-1;tot=0
 for x,y in ints:tot+=max(0,y-max(x,end));end=max(end,y)
 print(tag,round(100*tot/arc[-1],2),round(np.sqrt(np.mean(err**2))*1000,3),held[marked].all(),flush=True)
with ThreadPoolExecutor(4) as ex:list(ex.map(f,[(F,S,G) for F in [.025,.035,.045] for S in [.02,.04] for G in [.000025,.00008]]))
