"""Package compact evidence; retain large traces/checkpoints on the training host."""
import hashlib, json, tarfile
from pathlib import Path

base = Path(__file__).resolve().parent
files = [p for p in base.glob('*.json') if p.name != 'evidence_manifest.json']
files += list(base.glob('diagnostic_*/diagnostic_report.json'))
files += list(base.glob('ppo_short01/*_ppo.json'))
files += list(base.glob('ppo_short01/*agent.yaml'))
files += list(base.glob('ppo_short01/*snapshot.py'))
files += [p for p in base.glob('*.py')]
manifest = [{'path': str(p.relative_to(base)), 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]
(base / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
files.append(base / 'evidence_manifest.json')
with tarfile.open(base / 'results_bundle.tar.gz', 'w:gz') as bundle:
    for p in sorted(set(files)):
        bundle.add(p, arcname=str(p.relative_to(base)))
print('PACKAGED', len(files), 'files')
