"""Export recorded target-contact pairs; zero-force/non-target pairs are not included."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import csv,sys,numpy as np
OUT=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'))
tag=sys.argv[1]
a=np.load(OUT/f'{tag}.npz',allow_pickle=True)
with (OUT/f'{tag}_contacts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['time_s','phase','pair','force_N']);w.writeheader()
 for t,p,cs in zip(a['time'],a['phase'],a['contacts']):
  for c in cs['contact_geoms']:w.writerow(dict(time_s=float(t),phase=str(p),**c))
print(OUT/f'{tag}_contacts.csv')
