"""Report grasp changes separately from unchanged physical parameters."""
import json
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parent
baseline=json.loads((base/'run21/audit.json').read_text())
keys=['dt','kp','kd','contact_k','contact_d','torsion_radius','mass','hand_gravity','close_speed','force_target','orientation_weight','bounded_ik','freeze_after_grasp','adaptive_force','joint_pulse']
rows=[]
for folder in sorted(list(base.glob('long_side0*'))+list(base.glob('wrap0*'))):
 if not (folder/'report.json').exists():continue
 report=json.loads((folder/'report.json').read_text());audit=json.loads((folder/'audit.json').read_text());trace=json.loads((folder/'trace.json').read_text())
 changes={k:[baseline['args'][k],audit['args'][k]] for k in keys if baseline['args'][k]!=audit['args'][k]}
 invariant=not changes and baseline['source_sha256']==audit['source_sha256'] and all(np.allclose(baseline[k],audit[k]) for k in ['runtime_kp','runtime_kd'])
 hold=[x for x in trace if x['rise_m']>.015 and x['plate_force_N']<.05 and x['support_force_N']<.05]
 middle=np.array([x['middle_phalanx_force_N'] for x in hold if 'middle_phalanx_force_N' in x])
 rows.append({'run':folder.name,'unchanged_physics_and_drive_parameters':invariant,'physical_parameter_changes':changes,'reason':report['reason'],'lift_time_s':report['lift_time_s'],'max_free_hold_s':report['max_free_hold_s'],'max_lift_m':report['max_lift_m'],'middle_contact_fraction_during_airborne':((middle>.05).mean(0).tolist() if len(middle) else None),'middle_peak_N_during_airborne':(middle.max(0).tolist() if len(middle) else None),'final_slip_mm':trace[-1]['relative_slip_m']*1000})
(base/'wrap_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
