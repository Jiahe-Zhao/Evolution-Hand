import tarfile,json,hashlib
from pathlib import Path
base=Path(__file__).resolve().parent
paths=list(base.glob('*.py'))+list(base.glob('plans*.json'))+list(base.glob('*results.json'))+list(base.glob('*audit.json'))+list(base.glob('summary.json'))+list(base.glob('minimum_execution.json'))+list(base.glob('*.md'))+list(base.glob('candidate_profile.json'))
for directory in [*base.glob('run[0-9][0-9]'),base/'material_audit']:
 for name in ['audit.json','report.json','trace.json','controller_snapshot.py']:
  path=directory/name
  if path.exists():paths.append(path)
manifest={str(p.relative_to(base)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}
(base/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2));paths.append(base/'evidence_manifest.json')
with tarfile.open(base/'evidence.tar.gz','w:gz') as archive:
 for p in paths:archive.add(p,arcname=str(p.relative_to(base)))
print('evidence.tar.gz', (base/'evidence.tar.gz').stat().st_size,flush=True)
