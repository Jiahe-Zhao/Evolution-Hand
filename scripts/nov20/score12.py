
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np
from pathlib import Path
def assess(f):
 x=f['data'];a=int(f['start']);T=int(f['T']);S=int(f['S']);hz=240*S;ok=(x[:,13]>.5)&(x[:,8]>=.5)&(x[:,8]<=5)&(x[:,9]>.01)&(x[:,10]>.01)&(x[:,11]>=10)
 strokes=[];turns=[]
 # Locate the stone's actual extremum around each commanded reversal.
 ts=[a]
 for j in range(1,7):
  k=a+j*T;ids=np.arange(k-int(.12*hz),min(k+int(.12*hz)+1,len(x)));ts.append(int(ids[np.argmax(x[ids,1]*(-1)**(j-1))]))
 for j,(lo,hi) in enumerate(zip(ts,ts[1:])):
  m=ok[lo:hi+1];y=x[lo:hi+1,1]*(-1)**j;forward=np.diff(np.maximum.accumulate(y))*1000;good=m[:-1]&m[1:];dist=forward[good].sum()
  strokes.append(float(dist))
 for k in ts[1:-1]:
  w=x[k-round(.05*hz):k+round(.05*hz)+1];turns.append([w[:,12].max(),np.min(w[:,9:11]),w[:,11].min(),(w[:,8]<.001).sum()])
 return x,ok,ts,np.array(strokes),np.array(turns)
if __name__=='__main__':
 import sys
 for a in sys.argv[1:]:
  f=np.load(a);x,ok,ts,dist,tr=assess(f);print(Path(a).stem,dist.round(1),tr[:,0].round(2),tr[:,3])
