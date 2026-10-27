import subprocess,json,sys
from pathlib import Path
b=Path('/home/zjh/Evolution_PC/outputs/derl_pure_ppo_20261025')
out=b/'goal_full_20261027'
if out.exists():raise RuntimeError('Run directory already exists; refusing duplicate')
c=json.loads((b/'candidates.json').read_text())[0]
cmd=[sys.executable,'-u',str(b/'goal_worker.py'),'--source',c['source'],'--plans',c['plans'],'--output',str(out),'--num-envs','256','--steps','5000000','--eval-every','1048576','--seed','2027','--eval-seed','1062701','--headless']
with (b/'goal_full_20261027.log').open('x') as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(b/'goal_full_launch_20261027.json').write_text(json.dumps(dict(pid=p.pid,command=cmd),indent=2))
print(p.pid)
