import os,sys,json,subprocess
from pathlib import Path
base=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
for source,name,steps in [('strike_7_v2','strike_centered_v6',1200)]:
 name+=os.environ.get('CENTERED_RUN_SUFFIX','')
 if (base/'four_task_results'/name).exists():raise FileExistsError('Preserve existing results; set CENTERED_RUN_SUFFIX for a new run')
 d=json.load(open(base/'four_task_results'/source/'command.json'));cmd=d['args'];cmd[1]=str(base/'four_tasks/verify_centered_opposition.py');cmd[cmd.index('--output_dir')+1]=str(base/'four_task_results'/name);cmd+=['--max_steps',str(steps)]
 out=base/'four_task_results'/name;out.mkdir(exist_ok=True);env=dict(os.environ,**d['env'],BASELINE_RESET_SETTLE='1',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');(out/'command.json').write_text(json.dumps({'args':cmd,'env':d['env'],'reset_settle':True},indent=2))
 with (out/'process.log').open('w') as log:rc=subprocess.run(cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
 print(name,rc,flush=True)
