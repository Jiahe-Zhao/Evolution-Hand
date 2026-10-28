"""Sequential equal-budget Branch comparison on one GPU."""
import json,os,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parent
root=b/'branch_long_pair_20261028'
root.mkdir(exist_ok=False)
base=next(x['command'].copy() for x in json.loads((b/'scene_bottleneck_20261028/status.json').read_text()) if x['task']=='branch')
state={'started_unix':time.time(),'requested_steps_per_hand':10000000,'actual_steps_per_hand':9994240,'seed':2028,'runs':[]}
def save():
 p=root/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
for hand in ['ours','shadow']:
 cmd=base.copy();out=root/hand
 for flag,value in [('--output_dir',out),('--physics_batch_probe_envs',256),('--steps',10000000),('--eval_every',1048576),('--seed',2028),('--eval_seed',1062802)]:cmd[cmd.index(flag)+1]=str(value)
 env=os.environ.copy()
 for key in ['CONTACT_POLICY_AUDIT','CONTACT_AUDIT_MEAN_ONLY','CONTACT_DIAGNOSTIC','BRANCH_PLACEMENT_MANIFEST','PPO_PARENT_CHECKPOINT','GRAPH_TEACHER','BRANCH_DIAGNOSTIC_STEPS','BRANCH_NO_LOCK']:env.pop(key,None)
 env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',CONTACT_PROGRESS_REWARD='1',CONTACT_PPO_GAMMA='.999',GRAPH_DRIVEN_WRIST='0',BRANCH_COMPARISON_HAND=hand,BRANCH_COMPARISON_MODE='train',BRANCH_COMPARISON_MANIFEST=str(b/'branch_shadow_compare_20261028'/f'{hand}_shared_grid3.json'))
 row=dict(hand=hand,status='running',started_unix=time.time(),command=cmd,environment={k:v for k,v in env.items() if k.startswith(('CONTACT_','BRANCH_','GRAPH_','BASELINE_'))})
 with (root/(hand+'.log')).open('x') as log:
  proc=subprocess.Popen(['timeout','7200']+cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=env,stdout=log,stderr=subprocess.STDOUT)
  row['pid']=proc.pid;state['runs'].append(row);save();code=proc.wait()
 row.update(exit_code=code,finished_unix=time.time(),status='completed' if code==0 and (out/'completed.json').exists() else 'failed');save()
state['finished_unix']=time.time();save()
