
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from leaf_runtime import *
import json
p=json.loads((OUT_DIR/'plan_tipclear.json').read_text());h=LeafRuntime(side='left',wrist='six_axis',position=p['position'],orientation=p['orientation'],gravity=(0,0,-9.81));rows=[]
for k in range(8):
 h.advance(.001,dt=.000125);c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n];ss=c.soft_contact_shape.numpy()[:n];ff=h.out_f.numpy()[:n];dd=h.out_depth.numpy()[:n];groups={}
 for j in np.flatnonzero(np.linalg.norm(ff,axis=1)>.001):
  pid=int(ii[j,np.argmax(bb[j])]);region=h.region_names[h.regions[pid]] if 0<=pid<h.real_count and h.regions[pid]>=0 else 'other';label=region+' / '+str(h.model.shape_label[ss[j]]);g=groups.setdefault(label,[0.,0.,0]);g[0]+=float(np.linalg.norm(ff[j]));g[1]=max(g[1],float(dd[j]));g[2]+=1
 rows.append(dict(time=h.time,top=sorted(groups.items(),key=lambda x:x[1][0],reverse=True)[:10],min_particle_z=float(h.state.particle_q.numpy()[:h.real_count,2].min())))
(OUT_DIR/'initial_tipclear_contacts.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1],indent=2))
