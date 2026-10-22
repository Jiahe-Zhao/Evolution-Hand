"""Create a precise Git allowlist for this experiment, excluding large artifacts."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
names = {'single_update.json','single_update_status.json','single_update_analysis.json','report.json', 'before_ppo.json', 'after_ppo.json', 'bc_fit.json', 'dataset.json', 'execution.json', 'effective_agent.yaml', 'runtime_optimizer.json', 'training_diagnostics.json', 'training_metrics.json', 'checkpoint_hashes.json', 'kl_verification.json', 'collection.json', 'stopped.json', 'offline.json', 'feedback_gain.json', 'critic_warmup.json', 'residual_contract.json'}
files = [p for p in root.iterdir() if p.is_file() and p.suffix in {'.py', '.sh', '.md', '.json'} and p.name != 'archive_manifest.json']
files += [p for folder in root.iterdir() if folder.is_dir() for p in folder.iterdir() if p.is_file() and (p.name in names or p.name.endswith('_snapshot.py'))]
rows = [dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in sorted(set(files))]
(root / 'archive_manifest.json').write_text(json.dumps(rows, indent=2))
print(len(rows), 'files', sum(r['bytes'] for r in rows), 'bytes')
