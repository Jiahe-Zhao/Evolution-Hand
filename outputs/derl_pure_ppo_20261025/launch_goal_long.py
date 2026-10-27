"""One baseline, four independent pure-PPO tasks, each gated by a finite check."""
import os,sys,json,subprocess,time
from pathlib import Path
b=Path(__file__).resolve().parent
root=b/'goal_long_20261027';root.mkdir(exist_ok=False)
python=sys.executable
base_env=os.environ.copy();base_env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0')
source='/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/checks/baseline'
plans='/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/pilot_8_20261024/00_plans.json'
rows=[]
def save():
 p=root/'status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(rows,indent=2));t.replace(p)
def execute(task,check,benchmark=False):
 out=root/(task+'_benchmark' if benchmark else task+'_check' if check else task)
 env=base_env.copy();env['GRAPH_DRIVEN_WRIST']='1' if task in ('forage','strike') else '0';env['GRAPH_STRIKE_SUPPORT']='1' if task=='strike' else '0'
 if task=='grasp':
  cmd=[python,'-u',str(b/'goal_worker.py'),'--source',source+'/right/urdf/current_agent.urdf','--plans',plans,'--output',str(out),'--num-envs','16' if check else '256','--steps','10000000','--eval-every','1048576','--seed','2027','--eval-seed','1062801','--headless']
  if check:cmd.append('--limited-check')
 else:
  cmd=[python,'-u',str(b/'native_goal_worker.py'),'--task',task,'--output_dir',str(out),'--physics_batch_probe_envs','1' if check else '256','--steps','10000000','--eval_every','1048576','--seed','2027','--eval_seed','1062801','--headless','--baseline_genotype','/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/baseline_genotype.json','--mechanical_asset_dir',source,'--legacy_source_urdf','/home/zjh/Evolution_PC/Isaaclab_other/evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf']
  if check:cmd.append('--finite_check')
 if benchmark:
  cmd[cmd.index('--steps')+1]='32768';cmd.append('--benchmark')
 row=dict(task=task,phase='benchmark' if benchmark else 'check' if check else 'train',status='running',started=time.time(),command=cmd,output=str(out));rows.append(row)
 with (root/(out.name+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=env,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
 marker=out/('report.json' if benchmark else 'limited_check.json' if check else 'completed.json')
 valid=marker.exists() and (json.loads(marker.read_text()).get('passed',False) if check else True)
 row.update(exit_code=code,status='completed' if code==0 and valid else 'failed',duration_s=time.time()-row['started']);save()
 return row['status']=='completed'
# Tests run before any simulation or training. This excludes known reward/terminal bugs.
with (root/'unit_tests.log').open('x') as f:
 code=subprocess.run([python,'-m','unittest','test_goal_contract.py','test_native_goals.py'],cwd=b,stdout=f,stderr=subprocess.STDOUT).returncode
if code:raise RuntimeError('Goal contract tests failed')
passed=[]
for task in ('grasp','forage','branch','strike'):
 if execute(task,True):
  if task=='grasp' or execute(task,False,benchmark=True):passed.append(task)
# Do not interrupt the preceding baseline experiment. Treat a zombie as finished.
previous=Path('/proc/1046161/stat')
while previous.exists():
 try:state=previous.read_text().split(') ',1)[1].split()[0]
 except FileNotFoundError:break
 if state=='Z':break
 time.sleep(20)
for task in ('grasp','branch','forage','strike'):
 if task in passed:execute(task,False)
(root/'queue_completed.json').write_text(json.dumps({'passed_checks':passed,'all_tasks_completed':all(r['status']=='completed' for r in rows),'finished':time.time()},indent=2))
