"""Aggregate actual completion evidence; never infer success from process exit."""
import json
from pathlib import Path
b=Path(__file__).resolve().parent
rows=[]
for task,out in [('grasp',b/'goal_full_20261027')]+[(t,b/'native_goals_20261027_v2'/t) for t in ('branch','forage','strike')]:
 if task=='branch' and (b/'native_goals_20261027_v3/branch').exists():out=b/'native_goals_20261027_v3/branch'
 row=dict(task=task,output=str(out),status='not_started')
 if out.exists():
  row['status']='pending'
  for filename,key in [('progress.json','progress'),('effective_config.json','config'),('final_trials.json','final_trials'),('before_training.json','before')]:
   path=out/filename
   if path.exists():row[key]=json.loads(path.read_text())
  if (out/'completed.json').exists():row['status']='completed'
 rows.append(row)
(b/'goal_suite_status.json').write_text(json.dumps(rows,indent=2))
for r in rows:
 print(r['task'],r['status'],'frames',r.get('progress',{}).get('frames'),
       'final',[(x['action_mode'],x['successes'],x['num_envs']) for x in r.get('final_trials',[])])
