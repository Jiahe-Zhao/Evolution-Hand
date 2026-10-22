"""Merge only physically validated successful trajectories, retaining provenance."""
import argparse, hashlib, json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('inputs',nargs='+');a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError(out)
rows=[torch.load(path,map_location='cpu',weights_only=False) for path in a.inputs]
assert all(len(row['obs'])>0 and row['obs'].shape[1]==73 for row in rows)
provenance=[dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),samples=len(row['obs'])) for path,row in zip(a.inputs,rows)]
torch.save(dict(obs=torch.cat([r['obs'] for r in rows]),actions=torch.cat([r['actions'] for r in rows]),sources=provenance),out)
out.with_suffix('.json').write_text(json.dumps(provenance,indent=2))
print(provenance)
