"""Isolated finite/full-episode runs for the bounded-progress reward revision."""
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--task',choices=['branch','forage'],required=True);p.add_argument('--name',required=True);p.add_argument('--probe',action='store_true');p.add_argument('--sweep',action='store_true');p.add_argument('--steps',type=int,default=1048576);p.add_argument('--seed',type=int,default=2028);p.add_argument('--gamma',default='.999');p.add_argument('--envs',type=int,default=256);a=p.parse_args()
b=Path(__file__).resolve().parent;r=b/'contact_progress_20261028';r.mkdir(exist_ok=True);out=r/a.name
if out.exists():raise RuntimeError('Refuse to overwrite '+str(out))
prior=json.loads((b/'scene_bottleneck_20261028/status.json').read_text());cmd=next(x['command'].copy() for x in prior if x['task']==a.task)
for flag,value in [('--output_dir',out),('--physics_batch_probe_envs',1 if a.probe else a.envs),('--steps',a.steps),('--eval_every',a.steps),('--seed',a.seed)]:cmd[cmd.index(flag)+1]=str(value)
e=os.environ.copy()
for key in ['CONTACT_DIAGNOSTIC','BRANCH_PLACEMENT_MANIFEST','PPO_PARENT_CHECKPOINT','GRAPH_TEACHER','BRANCH_COMPARISON_HAND','BRANCH_COMPARISON_MODE','BRANCH_COMPARISON_MANIFEST']:e.pop(key,None)
e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',CONTACT_PROGRESS_REWARD='1',CONTACT_PPO_GAMMA=a.gamma,GRAPH_DRIVEN_WRIST='1' if a.task=='forage' else '0')
if a.task=='branch':e.update(BRANCH_COMPARISON_HAND='ours',BRANCH_COMPARISON_MODE='train',BRANCH_COMPARISON_MANIFEST=str(b/'branch_shadow_compare_20261028/ours_shared_grid3.json'))
if a.probe:
 if a.task!='forage':raise ValueError('Script probe is Forage only')
 e['CONTACT_DIAGNOSTIC']='progress_probe'
if a.sweep:
 if a.task!='forage' or a.envs!=16:raise ValueError('Sweep requires 16 Forage environments')
 e['CONTACT_DIAGNOSTIC']='sweep_probe'
log=(r/(a.name+'.log')).open('x');proc=subprocess.Popen(['timeout','3600']+cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/(a.name+'_launch.json')).write_text(json.dumps(dict(pid=proc.pid,command=cmd,task=a.task,seed=a.seed,gamma=a.gamma,probe=a.probe),indent=2));print(proc.pid)
