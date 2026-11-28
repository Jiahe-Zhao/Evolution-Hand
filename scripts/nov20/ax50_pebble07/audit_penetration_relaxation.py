"""Compare rigid collision distances using the actual runtime's exported MuJoCo model."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,subprocess,sys,numpy as np
P=Path(__file__).parent;O=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'))
s=(P/'diagnose_gap.py').read_text();s=s.replace("ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'","O/'penetration_runtime_model.mjb'");(P/'diagnose_runtime_gap.py').write_text(s)
z=np.load(O/'shape_topwide_v1.npz');records=[]
for mode in ['rigid_zero','full_zero','full_hold','full_posehold','full_graph']:
 key=f'penetration_{mode}_230';q=np.load(O/(key+'.npz'));tag=key+'_audit'
 np.savez_compressed(O/(tag+'.npz'),body_q=np.stack([z['body_q'][230],q['body_q']]),body_names=z['body_names'],time=[2.31,2.33])
 (O/(tag+'_result.json')).write_text(json.dumps(dict(object_xml=str(P/'assets/object_topwide_v1.xml'))))
 row=dict(mode=mode)
 for frame in [0,1]:
  subprocess.run([sys.executable,str(P/'diagnose_runtime_gap.py'),tag,str(frame)],stdout=subprocess.DEVNULL,check=True)
  r=json.loads((O/(tag+f'_frame{frame}_gap_diagnostic.json')).read_text())
  for label,suffix in [('thumb','link_1_2'),('index','link_2_3')]:
   val=min(x['distance_mm'] for x in r['pairs'] if x['body'].endswith(suffix) and (x['contype'] or x['conaffinity']))
   row[('start_' if frame==0 else 'end_')+label+'_rigid_gap_mm']=val
 records.append(row)
result=dict(scope='20ms diagnostic, recorded 2.31s pose restored with zero velocities/warm start; actual runtime rigid geometry; not original task replay',results=records)
(O/'penetration_relaxation_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
