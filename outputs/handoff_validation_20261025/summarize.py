"""Collect completed evidence without changing the training process."""
import json
from pathlib import Path
base=Path(__file__).resolve().parent
rows=[]
for name in ['control','standard','calibrated']:
 folder=base/name;file=folder/('control.json' if name=='control' else 'report.json')
 if not file.exists():continue
 data=json.loads(file.read_text());before=data['before'];after=data['repeat' if name=='control' else 'after']
 a,b=before['passed'],after['passed']
 row=dict(name=name,num_envs=before['num_envs'],before=before['successes'],after=after['successes'],
          lost_success=sum(x and not y for x,y in zip(a,b)),gained_success=sum(y and not x for x,y in zip(a,b)),
          before_falls=before['falls'],after_falls=after['falls'],
          bc_updates=data.get('bc_updates',0),after_diagnostics=after['diagnostics'])
 for artifact in ['inheritance.json','critic_calibration.json','policy_guard.json']:
  if (folder/artifact).exists():row[artifact[:-5]]=json.loads((folder/artifact).read_text())
 curve=folder/'learning_curve.jsonl'
 if curve.exists():
  records=[json.loads(line) for line in curve.read_text().splitlines()]
  row['last_epoch']=records[-1];row['epochs']=len(records)
 rows.append(row)
result=dict(scope='32 environments, same seed, complete 25-second episodes, short PPO',
            reward_and_morphology_unchanged=True,independent_seed_evaluation=False,
            observation_dimension=108,rows=rows)
(base/'results_summary.json').write_text(json.dumps(result,indent=2))
for row in rows:
 print(row['name'],row['before'],row['after'],'lost',row['lost_success'],'gained',row['gained_success'])
 if 'critic_calibration' in row:print('CRITIC',row['critic_calibration']['before'],row['critic_calibration']['after'])
 if 'policy_guard' in row:print('GUARD',row['policy_guard'])
