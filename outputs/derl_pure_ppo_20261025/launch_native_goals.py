"""Sequential native task integration validation; no BC, no inherited policy."""
import os,sys,json,subprocess,time,argparse
from pathlib import Path
b=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',choices=('branch','forage','strike'),default=['branch','forage','strike']);parser.add_argument('--wait-pid',type=int);a=parser.parse_args()
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}').exists():time.sleep(20)
root=b/'native_goals_20261027_v3';root.mkdir(exist_ok=False)
e=os.environ.copy();e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0')
rows=[]
for task in a.tasks:
 e['GRAPH_DRIVEN_WRIST']='1' if task in ('forage','strike') else '0';e['GRAPH_STRIKE_SUPPORT']='1' if task=='strike' else '0'
 out=root/task
 cmd=[sys.executable,'-u',str(b/'native_goal_worker.py'),'--task',task,'--output_dir',str(out),'--physics_batch_probe_envs','16','--steps','32768','--headless','--seed','2027','--baseline_genotype','/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/baseline_genotype.json','--mechanical_asset_dir','/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/checks/baseline','--legacy_source_urdf','/home/zjh/Evolution_PC/Isaaclab_other/evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf']
 row=dict(task=task,command=cmd,status='running',started=time.time());rows.append(row)
 with (root/(task+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=e,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid
  (root/'status.json').write_text(json.dumps(rows,indent=2));code=p.wait()
 row.update(exit_code=code,status='completed' if code==0 and (out/'completed.json').exists() else 'failed',duration_s=time.time()-row['started'])
 (root/'status.json').write_text(json.dumps(rows,indent=2))
