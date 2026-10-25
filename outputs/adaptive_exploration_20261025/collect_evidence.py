"""Collect small artifacts after the run; checkpoints stay on the remote host."""
import json,shutil,tarfile
from pathlib import Path
b=Path(__file__).resolve().parent;e=b/'evidence';e.mkdir(exist_ok=True)
for pattern in ('*.json','*.jsonl','*_snapshot.py','effective_agent.yaml'):
 for p in (b/'run').glob(pattern):shutil.copy2(p,e/p.name)
for name in ('tests.txt','status.json'):
 if (b/name).exists():shutil.copy2(b/name,e/name)
with tarfile.open('/tmp/adaptive_exploration_results_20261025.tgz','w:gz') as t:
 t.add(e,arcname='evidence');t.add(b/'summary.json',arcname='summary.json')
print('Evidence archived')
