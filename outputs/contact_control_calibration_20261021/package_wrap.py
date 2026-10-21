"""Package small validation artifacts; preserve full traces and videos on the host."""
import hashlib,json,tarfile
from pathlib import Path
base=Path(__file__).resolve().parent
files=[]
for pattern in ['plans_long_side*.json','plans_wrap*.json','wrap_summary.json','long_side_checks.json','WRAP_GRASP.md','wrap04_preview.mp4','wrap04_hold.png']:
 files.extend(base.glob(pattern))
traces=[]
for pattern in ['long_side0*','wrap0*']:
 for folder in base.glob(pattern):
  if not folder.is_dir():continue
  for name in ['report.json','audit.json','controller_snapshot.py']:
   if (folder/name).exists():files.append(folder/name)
  for name in ['trace.json','validation.mp4']:
   path=folder/name
   if path.exists():traces.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'kept_on_remote':True})
(base/'wrap_evidence_manifest.json').write_text(json.dumps(traces,indent=2));files.append(base/'wrap_evidence_manifest.json')
with tarfile.open(base/'wrap_delivery.tar.gz','w:gz') as archive:
 for path in sorted(set(files)):archive.add(path,arcname=str(path.relative_to(base)))
print('Packaged',len(set(files)),'files', (base/'wrap_delivery.tar.gz').stat().st_size,'bytes')
