
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import matplotlib,json,sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=OUT/'hao14';tag=sys.argv[1];a=np.load(o/(tag+'.npz'))['a'];curves=[np.array(x) for x in json.loads((o/(tag+'_curves.json')).read_text())];fig,ax=plt.subplots(1,2,figsize=(10,5),layout='constrained')
for k,c in enumerate(curves):
 ax[0].plot(c[:,0]*1000,c[:,1]*1000,label=str(k+1));ax[0].text(c[0,0]*1000,c[0,1]*1000,str(k+1));ax[1].plot(c[:,0]*1000,c[:,1]*1000,color='orange',ls='--')
ink=(a[:,9]>.01)&np.isfinite(a[:,7]);ax[1].plot(*np.where(ink[:,None],a[:,7:9]*1000,np.nan).T,lw=.6)
for b in ax:b.set_aspect('equal');b.grid(alpha=.2)
fig.savefig(o/(tag+'_overlay.png'),dpi=100)
