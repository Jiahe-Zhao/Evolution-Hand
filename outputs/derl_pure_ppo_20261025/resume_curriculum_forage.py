"""Retry Forage validation, then serialize its training after the original queue."""
import json,os,subprocess,sys,time
from pathlib import Path
from training_watchdog import wait_for_progress
b=Path(__file__).resolve().parent;root=b/'contact_curriculum_20261028'
rows=json.loads((root/'status.json').read_text());original=next(r for r in rows if r['label']=='forage_check')
cmd=original['command'].copy()
env=os.environ.copy();env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',GRAPH_DRIVEN_WRIST='1',GRAPH_STRIKE_SUPPORT='0')
state={}
def save():
 p=root/'forage_recovery_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
def run(label,n,steps):
 out=root/label
 cmd[cmd.index('--output_dir')+1]=str(out);cmd[cmd.index('--physics_batch_probe_envs')+1]=str(n);cmd[cmd.index('--steps')+1]=str(steps)
 state.update(status='running',label=label,started=time.time(),output=str(out));save()
 with (root/(label+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=env,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);state['pid']=p.pid;save()
  subprocess.run(['ionice','-c','2','-n','4','-p',str(p.pid)],check=False);code=wait_for_progress(p,out)
 state.update(status='completed' if code==0 and (out/'completed.json').exists() else 'failed',exit_code=code);save()
 return state['status']=='completed'
# Short diagnostic can share the GPU; long optimization remains serialized.
if run('forage_check_v2',32,4096):
 state.update(status='waiting_for_original_queue');save()
 while not (root/'queue_completed.json').exists():time.sleep(30)
 run('forage',256,10000000)
