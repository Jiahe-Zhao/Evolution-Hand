"""Preflight then our mechanical hand; Shadow remains explicitly pending validation."""
import os,json,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'strike_50m_pair_20261029';root.mkdir(exist_ok=False)
state={'budget_per_hand':50000000,'order':['ours','shadow'],'shadow_status':'pending_wrist_adapter_and_validation','runs':[]}
def save():
 p=root/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
base=next(x['command'].copy() for x in json.loads((b/'scene_bottleneck_20261028/status.json').read_text()) if x['task']=='strike')
e=os.environ.copy()
for k in list(e):
 if k.startswith(('CONTACT_','BRANCH_','FORAGE_','STRIKE_','GRAPH_','EVOLUTION_STRIKE_')) or k in ['PPO_PARENT_CHECKPOINT']:e.pop(k,None)
e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_PPO_GAMMA='.999',GRAPH_DRIVEN_WRIST='1',GRAPH_STRIKE_SUPPORT='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0',STRIKE_PROGRESS_REWARD='1')
for label,n,steps in [('ours_preflight',32,4096),('ours',256,50000000)]:
 out=root/label;cmd=base.copy()
 for flag,value in [('--output_dir',out),('--physics_batch_probe_envs',n),('--steps',steps),('--eval_every',2097152),('--seed',2029),('--eval_seed',1062902)]:cmd[cmd.index(flag)+1]=str(value)
 cmd+=['--whole_hand_tactile']
 row=dict(label=label,command=cmd,environment={k:v for k,v in e.items() if k.startswith(('CONTACT_','STRIKE_','GRAPH_','EVOLUTION_STRIKE_','BASELINE_'))},started_unix=time.time(),status='running');state['runs'].append(row)
 with (root/(label+'.log')).open('w') as f:
  p=subprocess.Popen(['timeout','28800']+cmd,env=e,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
 row.update(exit_code=code,finished_unix=time.time(),status='completed' if code==0 and (out/'completed.json').exists() else 'failed');save()
 if row['status']!='completed':raise RuntimeError(label+' failed; do not continue')
state['ours_finished_unix']=time.time();save()
