"""Export recorded target-contact pairs; zero-force/non-target pairs are not included."""
from pathlib import Path
import csv,sys,numpy as np
OUT=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07')
tag=sys.argv[1]
a=np.load(OUT/f'{tag}.npz',allow_pickle=True)
with (OUT/f'{tag}_contacts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['time_s','phase','pair','force_N']);w.writeheader()
 for t,p,cs in zip(a['time'],a['phase'],a['contacts']):
  for c in cs['contact_geoms']:w.writerow(dict(time_s=float(t),phase=str(p),**c))
print(OUT/f'{tag}_contacts.csv')
