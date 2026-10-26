import json,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'reevaluation_9_20261026';root.mkdir(exist_ok=False)
original=json.loads((b/'runs/status.json').read_text());rows=[]
assert len(original)==9 and all(r['status']=='completed' for r in original)
for old in original:
 name=old['name'];c=old['candidate'];out=root/name;checkpoint=b/'runs'/name/'final.pth'
 cmd=[sys.executable,'-u',str(b/'reevaluate_worker.py'),'--source',c['source'],'--plans',c['plans'],'--output',str(out),'--checkpoint',str(checkpoint),'--num-envs','256','--headless']
 row=dict(name=name,status='running',command=cmd,started=time.time(),original_mean_successes=old.get('successes'));rows.append(row)
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;(root/'status.json').write_text(json.dumps(rows,indent=2));code=p.wait()
 row.update(status='completed' if code==0 and (out/'completed.json').exists() else 'failed',exit_code=code,duration_s=time.time()-row['started'])
 (root/'status.json').write_text(json.dumps(rows,indent=2))
 subprocess.run([sys.executable,str(b/'summarize_reevaluation.py'),str(root)],check=True)
 print('FINISHED',name,row['status'],flush=True)
repo=b.parent.parent
paths=[str((root/x).relative_to(repo)) for x in ('RESULTS.md','comparison.json','evidence')]
subprocess.run(['git','add','-f','--',*paths],cwd=repo,check=True)
subprocess.run(['git','commit','--only','-m','Record nine-checkpoint stochastic reevaluation','--',*paths],cwd=repo,check=True)
print('ALL_REEVALUATIONS_FINISHED',flush=True)
