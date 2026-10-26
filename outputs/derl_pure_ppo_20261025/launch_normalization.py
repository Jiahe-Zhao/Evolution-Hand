import json,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'normalization_ab_20261026';root.mkdir(exist_ok=False)
c=json.loads((b/'candidates.json').read_text())[1];rows=[]
for freeze in (True,False):
 name='frozen' if freeze else 'adaptive';out=root/name
 cmd=[sys.executable,'-u',str(b/'normalization_worker.py'),'--source',c['source'],'--plans',c['plans'],'--output',str(out),'--resume',str(b/'runs/thumb_lateral_plus_s2027/best.pth'),'--seed','2027','--eval-seed','1062601','--num-envs','256','--steps','950272','--eval-every','475136','--headless']
 if freeze:cmd.append('--freeze-observation')
 row=dict(name=name,status='running',command=cmd,started=time.time());rows.append(row)
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;(root/'status.json').write_text(json.dumps(rows,indent=2));code=p.wait()
 row.update(status='completed' if code==0 and (out/'report.json').exists() else 'failed',exit_code=code,duration_s=time.time()-row['started'])
 if (out/'report.json').exists():
  r=json.loads((out/'report.json').read_text());row.update(before=r['before']['successes'],after=r['after']['successes'],normalization_check=r['normalization_check'])
 (root/'status.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
print('AB_COMPLETED',flush=True)
