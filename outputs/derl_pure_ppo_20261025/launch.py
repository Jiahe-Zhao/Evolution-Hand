"""Sequential equal-budget jobs, round-robin over morphologies then seeds."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--num-envs',type=int,default=256);p.add_argument('--seeds',default='2027,2028,2029');p.add_argument('--steps',type=int,default=5_000_000);a=p.parse_args()
candidates=json.loads((b/'candidates.json').read_text());seeds=[2027] if a.smoke else [int(s) for s in a.seeds.split(',')]
root=b/('smoke' if a.smoke else 'runs');root.mkdir(exist_ok=False)
state=[]
for seed in seeds:
 for c in candidates:
  label=f'{c["name"]}_s{seed}';out=root/label
  budget=a.num_envs*128*2 if a.smoke else a.steps
  cmd=[sys.executable,'-u',str(b/'worker.py'),'--source',c['source'],'--plans',c['plans'],'--output',str(out),'--seed',str(seed),'--num-envs',str(a.num_envs),'--steps',str(budget),'--headless']
  if a.smoke:cmd.append('--smoke')
  row=dict(name=label,status='running',started=time.time(),command=cmd,candidate=c);state.append(row)
  (root/'status.json').write_text(json.dumps(state,indent=2))
  with (root/(label+'.log')).open('w') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);row['pid']=proc.pid
   (root/'status.json').write_text(json.dumps(state,indent=2));code=proc.wait()
  row.update(status='completed' if code==0 and (out/'report.json').exists() else 'failed',exit_code=code,duration_s=time.time()-row['started'])
  if (out/'report.json').exists():
   r=json.loads((out/'report.json').read_text());row.update(successes=r['after']['successes'],episodes=r['after']['num_envs'])
  (root/'status.json').write_text(json.dumps(state,indent=2))
  if not a.smoke:subprocess.run([sys.executable,str(b/'summarize.py')],check=False)
  # One morphology's failure is recorded; the other requested morphologies still run.
if not a.smoke:
 subprocess.run([sys.executable,str(b/'summarize.py')],check=False)
 repo=b.parent.parent
 paths=[str((b/x).relative_to(repo)) for x in ('RESULTS.md','results.json','evidence')]
 subprocess.run(['git','-C',str(repo),'add','-f','--',*paths],check=True)
 subprocess.run(['git','-C',str(repo),'commit','--only','-m','Record DERL-style pure PPO validation outcomes','--',*paths],check=True)
print('ALL_JOBS_FINISHED',flush=True)
