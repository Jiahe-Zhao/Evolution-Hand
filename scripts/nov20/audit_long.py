
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import csv,collections
o=OUT/'pen140';a=np.load(o/'write.npz')['a'];h=np.load(o/'hover.npz')['a'];print('hold',a[:,10:12].min(0),'hover',h[:,9].max(),h[:,6].min()-np.loadtxt(o/'hover_origin.txt')[3]-.001,'hover end err',np.linalg.norm(h[-1,4:6]-h[-1,2:4]),'hover grip',h[h[:,1]>0,10:12].min(0))
loads=collections.defaultdict(lambda:np.zeros(len(a)));axial=collections.defaultdict(list);gaps=collections.defaultdict(list)
for r in csv.DictReader(open(o/'write_rear.csv')):
 b=r['body'];seg=r['rod_segment'];f=float(r['force_N']);t=round(float(r['time_s'])*960)
 if f>.001 and b.startswith('link_'):
  if b in ('link_1_3','link_2_3','link_3_3') and a[t,1]>0:axial[b].append(float(r['from_nib_mm']))
  if seg=='rear':loads[b][t]+=f;gaps[b].append(float(r['gap_m']))
for b,val in loads.items():print('rear',b,'loaded %',round(100*(val>.001).mean(),1),'peak',val.max(),'median axial grip',np.median(axial[b]) if axial[b] else None,'penetration mm',-min(gaps[b])*1000)
