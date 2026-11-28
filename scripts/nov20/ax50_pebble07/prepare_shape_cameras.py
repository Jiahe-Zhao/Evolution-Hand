"""Shared close views that contain both variants' recorded stones and active fingertips."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from render_pebble import *
from scipy.spatial.transform import Rotation
O=OUT
points=[]
for tag in ['shape_original_v1','shape_topwide_v1']:
 z=np.load(O/(tag+'.npz'));bq=z['body_q'];names=[str(n).rsplit('/',1)[-1] for n in z['body_names']];idx={n:i for i,n in enumerate(names)}
 meta=json.loads((O/(tag+'_result.json')).read_text());geoms,_=load_object(meta['object_xml'])
 eligible=np.flatnonzero(z['time']>=1.8);frames=eligible[np.unique(np.linspace(0,len(eligible)-1,50,dtype=int))]
 for i in frames:
  rr=rot(bq[i,:,3:]);p=bq[i,:,:3]
  for g in geoms:
   j=idx[g['body']];v=g['v']@(rr[j]@g['R']).T+p[j]+rr[j]@g['p'];points.extend(v)
  for name,local in [('link_1_2',[0,.0036,.0235]),('link_2_3',[0,.0031,.0195])]:
   j=idx[name];tip=p[j]+rr[j]@np.array(local)
   points.extend(tip+np.array([[x,y,z] for x in [-.007,.007] for y in [-.007,.007] for z in [-.007,.007]]))
old=json.loads((O/'shape_comparison_cameras.json').read_text());cams=[(np.array(old[0][0]),old[0][1],np.array(old[0][2])),camera(points,[1,-.8,.3],.84),camera(points,[1,.08,.16],.84)]
(O/'shape_comparison_cameras_reviewed.json').write_text(json.dumps([[b.tolist(),float(s),o.tolist()] for b,s,o in cams],indent=2))
print('shared cameras include full recorded objects and fingertips')
