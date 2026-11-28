
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation as R
OUT=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v3.json').read_text());close=json.loads((Path(__file__).parent/'candidates/pinch_close_candidate_v3.json').read_text());tilt=R.from_euler('y',.45);pivot=np.array([-.093,.003,.0345]);c['position']=(pivot+tilt.apply(np.array(c['position'])-pivot)).tolist();c['orientation']=(tilt*R.from_quat(c['orientation'])).as_quat().tolist()
for k in [4,8,12]:c['joint_pose'][k:k+4]=[0,0,0,0]
c['world_tilt_rad']=.45;c['world_tilt_pivot']=pivot.tolist();close['wrist_delta_world']=tilt.apply(close['wrist_delta_world']).tolist()
for name,data in [('pinch_candidate',c),('pinch_close_candidate',close)]:
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2));(OUT/(name+'_v4.json')).write_text(json.dumps(data,indent=2))
print('tilt',c['world_tilt_rad'],'position',c['position'])
