"""Sequential single-GPU comparison jobs with explicit completion gates."""
import json,subprocess,time,sys,os
from pathlib import Path
b=Path(__file__).resolve().parent;r=b/'branch_shadow_compare_20261028'
plan_path=Path(sys.argv[1]);plan=json.loads(plan_path.read_text());status_path=plan_path.with_suffix('.status.json')
status=dict(started=time.time(),plan=str(plan_path),jobs=[])
def save():
 p=status_path.with_suffix('.tmp');p.write_text(json.dumps(status,indent=2));p.replace(status_path)
for job in plan:
 command=[sys.executable,str(b/'launch_branch_comparison.py')]
 for key,value in job.items():command.extend(['--'+key,str(value)])
 row=dict(job=job,started=time.time(),state='starting');status['jobs'].append(row);save()
 launched=subprocess.run(command,capture_output=True,text=True)
 row['launcher_output']=launched.stdout;row['launcher_error']=launched.stderr
 if launched.returncode:row['state']='launch_failed';save();sys.exit(1)
 metadata=json.loads((r/(job['name']+'_launch.json')).read_text());pid=metadata['pid'];row.update(pid=pid,state='running');save()
 while True:
  if (r/job['name']/'completed.json').exists():
   row.update(state='completed',finished=time.time());save();break
  try:state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]
  except FileNotFoundError:state='gone'
  if state in ('Z','gone'):
   row.update(state='failed',finished=time.time());save();sys.exit(1)
  time.sleep(10)
status.update(state='completed',finished=time.time());save()
