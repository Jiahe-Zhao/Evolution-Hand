"""Serialized baseline curriculum, full-episode checks, explicit stage gate."""
import json,os,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
root=b/'contact_curriculum_20261028';root.mkdir(exist_ok=False)
env=os.environ.copy();env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0',CONTACT_CURRICULUM='1')
rows=[]
def save():
 p=root/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(p)
def execute(task,label,steps=10000000,n=256,stage=1,parent=None):
 out=root/label;e=env.copy();e.update(GRAPH_DRIVEN_WRIST='1' if task in ('forage','strike') else '0',GRAPH_STRIKE_SUPPORT='1' if task=='strike' else '0',STRIKE_TRAIN_STAGE=str(stage))
 if parent:e['PPO_PARENT_CHECKPOINT']=str(parent)
 cmd=[sys.executable,'-u',str(b/'native_goal_worker.py'),'--task',task,'--output_dir',str(out),'--physics_batch_probe_envs',str(n),'--steps',str(steps),'--eval_every','1048576','--seed','2028','--eval_seed','1062802','--headless','--baseline_genotype','/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/baseline_genotype.json','--mechanical_asset_dir','/home/zjh/Evolution_PC/outputs/discrete_hand_evolution_20261024/checks/baseline','--legacy_source_urdf','/home/zjh/Evolution_PC/Isaaclab_other/evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf']
 row=dict(task=task,label=label,status='running',started=time.time(),command=cmd,output=str(out));rows.append(row)
 with (root/(label+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=e,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;save()
  subprocess.run(['ionice','-c','2','-n','4','-p',str(p.pid)],check=False)
  code=p.wait()
 valid=code==0 and (out/'completed.json').exists()
 row.update(status='completed' if valid else 'failed',exit_code=code,elapsed_s=time.time()-row['started']);save()
 return valid
# Grasp's successful policy remains the control baseline. No redundant retrain.
passed=[]
for task in ('strike','branch','forage'):
 if execute(task,task+'_check',4096,32):passed.append(task)
# Each native task gets a fresh policy; no use of the defective old Forage run.
for task in ('branch','forage','strike'):
 if task in passed:execute(task,task+'_stage1' if task=='strike' else task)
# Promote only the evaluated final policy; do not silently load an untested best.
p=root/'strike_stage1/final_trials.json'
if p.exists():
 trials=json.loads(p.read_text());rates=[r['successes']/r['num_envs'] for r in trials]
 approved=len(trials)>=3 and all(rate>=(.70 if row['action_mode']=='mean' else .50) for row,rate in zip(trials,rates))
 (root/'strike_promotion.json').write_text(json.dumps(dict(approved=approved,rates=rates,rule='two action seeds >=50%, mean >=70%; stage1 lift/hold only'),indent=2))
 if approved:execute('strike','strike_stage2',stage=2,parent=root/'strike_stage1/final.pth')
(root/'queue_completed.json').write_text(json.dumps(dict(finished=time.time(),checks_passed=passed,rows=rows),indent=2))
