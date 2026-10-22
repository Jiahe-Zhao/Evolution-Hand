"""Summarize completed runs without treating missing results as failures/successes."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for p in sorted(root.glob('*/report.json')):
 d=json.loads(p.read_text())
 if 'before' not in d:continue
 for stage in ('before','after'):
  r=d.get(stage)
  if not r:continue
  diag=r['diagnostics']
  rows.append(dict(run=p.parent.name,stage=stage,successes=r['successes'],num_envs=r['num_envs'],falls=r['falls'],max_speed_m_s=diag['max_speed_m_s'],max_digit_force_N=diag['max_digit_force_N'],max_limit_rad=diag['max_limit_rad']))
(root/'results_summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
