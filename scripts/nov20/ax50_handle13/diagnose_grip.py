"""Inspect recorded bail position relative to fingers; never modify simulation."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
OUT=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_handle13'))
tag=sys.argv[1]
z=np.load(OUT/(tag+'.npz'));names=[str(n).rsplit('/',1)[-1] for n in z['body_names']];poses=z['body_q'];times=z['time'];trace=list(csv.DictReader((OUT/(tag+'_trace.csv')).open()))
# Original grip center, unchanged in the imported object.
grip_local=np.array([.064,0.,0.]);b=names.index('bail')
grip=poses[:,b,:3]+Rotation.from_quat(poses[:,b,3:]).apply(np.tile(grip_local,(len(times),1)))
parts=['link_3_2','link_3_3','link_4_2','link_4_3'];rows=[]
for t in [1.6,1.7,2.4,2.8,3.0,3.1,3.2,3.3,3.4,3.5,3.65,3.8,4.0,4.3]:
 if t>times[-1]+.001:continue
 i=int(np.argmin(abs(times-t)));rel={}
 for name in parts:
  j=names.index(name);d=grip[i]-poses[i,j,:3];rel[name]={'world_mm':(d*1000).tolist(),'bone_local_mm':(Rotation.from_quat(poses[i,j,3:]).inv().apply(d)*1000).tolist()}
 rows.append({'time_s':float(times[i]),'phase':trace[i]['phase'],'grip_world_m':grip[i].tolist(),'relative_to_bone_origin':rel,'bail_contact_N':float(trace[i]['bail_contact_N']),'lift_mm':float(trace[i]['lift_m'])*1000,'tilt_deg':float(trace[i]['tilt_deg'])})
result={'scope':'Recorded poses only; relative to bone origins, not surface penetration or acceptance proof. Body names avoid MuJoCo/Newton index offset.','samples':rows}
(OUT/(tag+'_grip_diagnosis.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
