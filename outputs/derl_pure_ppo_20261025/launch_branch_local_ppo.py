"""Short pure PPO after independent perturbed script feasibility validation."""
import json,os,subprocess,sys,time
from pathlib import Path
from training_watchdog import wait_for_progress
b=Path(__file__).resolve().parent;root=b/'branch_local_ppo_v3_20261028'
root.mkdir(exist_ok=False)
validation=b/'branch_local_guarded_20261028'
# Wait only for the named diagnostic, with a bounded deadline.
deadline=time.time()+400
while not (validation/'completed.json').exists():
 if time.time()>deadline:raise RuntimeError('Perturbation validation did not finish')
 time.sleep(5)
trials=json.loads((validation/'script_validation.json').read_text())
passed=all(r['sustained_3s']/r['episodes']>=.8 and r['initial_collision_envs']==0 and r['initial_successes']==0 and r['released_clear']==r['episodes'] and min(r['first_fresh_contact_frame'])>=0 for r in trials)
(root/'gate.json').write_text(json.dumps(dict(passed=passed,trials=trials,rule='each seed >=80% sustained 3 seconds; no initial sustained contact or initial success; all environments recover fresh reports and release clear'),indent=2))
if not passed:
 (root/'status.json').write_text(json.dumps(dict(status='blocked_by_validation')));sys.exit(0)
prior=json.loads((b/'scene_bottleneck_20261028/status.json').read_text())[1];cmd=prior['command'].copy()
cmd[cmd.index('--output_dir')+1]=str(root/'train');cmd[cmd.index('--physics_batch_probe_envs')+1]='256';cmd[cmd.index('--steps')+1]='524288';cmd[cmd.index('--eval_every')+1]='524288'
env=os.environ.copy();env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',GRAPH_DRIVEN_WRIST='0',GRAPH_STRIKE_SUPPORT='0',BRANCH_PLACEMENT_MANIFEST=str(b/'branch_translation_20261028/candidate57.json'),BRANCH_POSITION_JITTER_M='0.002')
env.pop('CONTACT_DIAGNOSTIC',None);env.pop('PPO_PARENT_CHECKPOINT',None)
with (root/'train.log').open('x') as f:
 p=subprocess.Popen(cmd,env=env,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT)
 (root/'status.json').write_text(json.dumps(dict(status='running',pid=p.pid,started=time.time(),command=cmd),indent=2))
 subprocess.run(['ionice','-c','2','-n','4','-p',str(p.pid)])
 code=wait_for_progress(p,root/'train')
(root/'status.json').write_text(json.dumps(dict(status='completed' if code==0 and (root/'train/completed.json').exists() else 'failed',exit_code=code,finished=time.time()),indent=2))
