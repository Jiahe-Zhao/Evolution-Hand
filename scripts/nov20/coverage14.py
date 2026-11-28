
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
exec(open(__file__.replace('coverage14','report14')).read().split('assert not ink')[0])
ix=np.flatnonzero(ok);s=arc[idx];ints=sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ix[j+1]-ix[j]==1);end=-1;tot=0
for x,y in ints:tot+=max(0,y-max(x,end));end=max(end,y)
print('uninterrupted loaded arc coverage',tot/arc[-1],'on path proportion',ok[(a[:,1]>0)&(a[:,1]<1)].mean(),'hover clearance',h[:,6].min()-(pos[2]-.0334))
