"""Package reproducibility artifacts; exclude datasets, weights, USD and caches."""
from pathlib import Path
import argparse,tarfile
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
root=Path(__file__).resolve().parent
names={'report.json','before_ppo.json','after_ppo.json','bc_fit.json','dataset.json','execution.json','effective_agent.yaml','runtime_optimizer.json','training_diagnostics.json','training_metrics.json','checkpoint_hashes.json','kl_verification.json','collection.json','stopped.json','controller_snapshot.py','expert_feedback_snapshot.py','retrieval_bc_snapshot.py','ppo_numerics_snapshot.py','bc_anchor_snapshot.py','offline.json','feedback_gain.json','critic_warmup.json','critic_warmup_snapshot.py','residual_contract.json'}
files=[p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.md','.json'}]
files += [p for directory in root.iterdir() if directory.is_dir() for p in directory.iterdir() if p.is_file() and (p.name in names or p.name.endswith("_snapshot.py"))]
with tarfile.open(a.output,'w:gz') as archive:
 for path in sorted(set(files)):archive.add(path,arcname=str(path.relative_to(root)))
print(len(files),'files')
