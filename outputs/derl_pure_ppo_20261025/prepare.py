"""Select previously authorized ternary phenotypes; no shape edits or BC loading."""
import hashlib,json
from pathlib import Path
b=Path(__file__).resolve().parent
prior=b.parent/'discrete_hand_evolution_20261024/pilot_8_20261024'
rows=[]
for old_id,label in [('00','baseline'),('03','thumb_lateral_plus'),('06','thumb_length_plus')]:
 job=json.loads((prior/(old_id+'_job.json')).read_text());candidate=json.loads(Path(job['candidate']).read_text())
 assert candidate['passed']
 urdf=Path(candidate['artifacts']['right']);plans=prior/(old_id+'_plans.json')
 rows.append(dict(name=label,genes=job['code'],source=str(urdf),plans=str(plans),source_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(plans.read_bytes()).hexdigest(),initialization='morphology-specific static collision-free pregrasp; no teacher actions',
                  parent_checkpoint=None,bc_updates=0))
(b/'candidates.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
