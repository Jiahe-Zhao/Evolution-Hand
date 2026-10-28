"""Launch one isolated comparison run; never overwrite an existing output."""
import argparse,os,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--hand',choices=['ours','shadow'],required=True);p.add_argument('--mode',default='inspect');p.add_argument('--name',required=True);p.add_argument('--envs',type=int,default=32);p.add_argument('--steps',type=int,default=1048576);p.add_argument('--seed',type=int,default=2028);p.add_argument('--manifest');a=p.parse_args()
b=Path(__file__).resolve().parent;root=b/'branch_shadow_compare_20261028';root.mkdir(exist_ok=True);out=root/a.name
if out.exists():raise RuntimeError('Refuse to overwrite '+str(out))
c=json.loads((b/'scene_bottleneck_20261028/status.json').read_text())[1]['command'].copy()
for flag,val in [('--output_dir',out),('--physics_batch_probe_envs',a.envs),('--steps',a.steps),('--eval_every',a.steps),('--seed',a.seed)]:c[c.index(flag)+1]=str(val)
e=os.environ.copy()
for key in ['BRANCH_PLACEMENT_MANIFEST','CONTACT_DIAGNOSTIC','PPO_PARENT_CHECKPOINT','GRAPH_TEACHER','BRANCH_DIAGNOSTIC_STEPS','BRANCH_NO_LOCK']:e.pop(key,None)
e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BRANCH_COMPARISON_HAND=a.hand,BRANCH_COMPARISON_MODE=a.mode,CONTACT_CURRICULUM='1')
if a.manifest:e['BRANCH_COMPARISON_MANIFEST']=a.manifest
f=(root/(a.name+'.log')).open('x')
p=subprocess.Popen(['timeout','3600']+c,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(root/(a.name+'_launch.json')).write_text(json.dumps(dict(pid=p.pid,command=c,variant=a.hand,mode=a.mode,seed=a.seed),indent=2));print(p.pid)
