import json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
b=Path(__file__).resolve().parent
cal=b.parent/'contact_control_calibration_20261021'
a=json.loads((cal/'wrap04/audit.json').read_text())
r=json.loads((b/'audit_replay01/trace.json').read_text())
t=json.loads((cal/'wrap04/trace.json').read_text())
R=np.array(a['plan']['rotation']).T@Rotation.from_euler('y',np.pi).as_matrix()
rows=[]
for k in [0,45,120,300,600,900,1199]:
 x=r[k];target=np.array(x['target']);ref=np.array(t[k]['target']);actual=np.array(x['obj']);force=np.linalg.norm(np.array(x['force']),axis=-1)
 rows.append(dict(t_s=k/60,ages=x['age'],same_episode_as_teacher=[v==k+1 for v in x['age']],replay_target_error_rad=float(np.abs(target[:8,3:]-ref[3:]).max()),bc_target_error_rad=float(np.abs(target[8:,3:]-ref[3:]).max()),bc_wrist_target_error_mm=float(np.linalg.norm((target[8:,:3]-ref[:3])@R.T,axis=1).max()*1000),replay_object_height_m=float(actual[:8,2].mean()),bc_object_height_m=float(actual[8:,2].mean()),bc_peak_digit_force_N=float(force[8:].max())))
out=dict(samples=rows,offline=json.loads((b/'audit_replay01/offline.json').read_text()),replay_report=json.loads((b/'audit_replay01/report.json').read_text()))
(b/'old_failure_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(rows,indent=2))
