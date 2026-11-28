
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
exec(open(__file__.replace('diag14','report14')).read().split('assert not ink')[0]);print('cov/rms/held/hover_min',cov,rms,held[marked].mean(),h[:,6].min()-(pos[2]-.0354),'hoverink',ink(h).sum(),'gripmin',a[marked,10:12].min(axis=0),'missing',np.setdiff1d(np.arange(100),bins));print(a[::240,[1,4,5,6,9,10,11]])
