
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from leaf_runtime import *
from scipy.spatial.transform import Rotation as R
import json
p=json.loads((OUT_DIR/'plan_tipclear.json').read_text());h=LeafRuntime(side='left',wrist='six_axis',position=p['position'],orientation=p['orientation'],gravity=(0,0,-9.81));pp=h.state.particle_q.numpy()[:h.real_count];b=h.state.body_q.numpy();region=h.region_names.index('index_distal');x=pp[np.array(h.regions)==region];bi=h.body_names.index('link_2_3');local=R.from_quat(b[bi,3:]).inv().apply(x-b[bi,:3]);n=np.array([0,np.cos(np.deg2rad(20)),np.sin(np.deg2rad(20))]);support=local@ n;imax=int(support.argmax());r=dict(scope='rest soft surface; not deformed contact',count=len(local),min_local_m=local.min(0).tolist(),max_local_m=local.max(0).tolist(),support_vertex_m=local[imax].tolist(),support_value_m=float(support[imax]),planned_point_m=p['tip_local'][0],planned_support_m=float(np.array(p['tip_local'][0])@n));(OUT_DIR/'tip_surface.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
