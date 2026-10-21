import json,hashlib
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parent;folder=base/'ppo_short01';r=json.loads((folder/'ppo_report.json').read_text());baseline=json.loads((base.parent/'contact_control_calibration_20261021/wrap04/audit.json').read_text())
fields=['role','static_friction','dynamic_friction','compliant_stiffness','compliant_damping']
def materials(rows):return sorted(set(tuple(x.get(k) for k in fields) for x in rows),key=str)
summary={k:v for k,v in r.items() if k!='materials'}
summary['material_signatures']=materials(r['materials']);summary['baseline_material_signatures']=materials(baseline['materials'])
summary['matches_baseline_source_and_drives']=r['source_sha256']==baseline['source_sha256'] and np.allclose(r['runtime_kp'],baseline['runtime_kp'][0]) and np.allclose(r['runtime_kd'],baseline['runtime_kd'][0])
summary['matches_baseline_materials']=materials(r['materials'])==materials(baseline['materials'])
summary['checkpoints']={p.name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in folder.glob('*.pth')}
(base/'ppo_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
