"""Archive small reproducibility artifacts; leave checkpoints on the training host."""
import hashlib,json,shutil
from pathlib import Path
base=Path(__file__).resolve().parent
evidence=base/'evidence';evidence.mkdir(exist_ok=True)
for name in ('control','standard','calibrated'):
    source=base/name
    dest=evidence/name;dest.mkdir(exist_ok=True)
    for pattern in ('*.json','*.jsonl','effective_agent.yaml','controller_snapshot.py','*_snapshot.py'):
        for path in source.glob(pattern):shutil.copy2(path,dest/path.name)
for name in ('status.json','tests.txt'):
    if (base/name).exists():shutil.copy2(base/name,evidence/name)
manifest={}
for path in sorted(base.glob('*.py')):
    manifest[str(path.relative_to(base))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in [base.parent/'wrap_learning_repair_20261022/critic_warmup.py',base.parent/'wrap_learning_repair_20261022/ppo_numerics.py']:
    if path.exists():
        shutil.copy2(path,evidence/path.name)
        manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(evidence/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Evidence collected',len(list(evidence.rglob('*'))),'entries')
