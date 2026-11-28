
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_keys17'));a=np.genfromtxt(p/'object_probe.csv',delimiter=',',names=True)
f,ax=plt.subplots(figsize=(10,4))
for k in range(1,5):ax.plot(a['time_s'],a[f'key{k}_m']*1000,label=f'K{k}')
for y,label in [(2,'trigger'),(.5,'released'),(4,'nominal stop')]:ax.axhline(y,ls='--',lw=.8,color='gray');ax.text(2.4,y,label,ha='right',va='bottom')
ax.set(xlabel='Time (s)',ylabel='Travel (mm)',title='Object-only diagnostic: 0.72 N step per key (not hand acceptance)');ax.legend();f.tight_layout();f.savefig(p/'object_probe.png',dpi=160)
