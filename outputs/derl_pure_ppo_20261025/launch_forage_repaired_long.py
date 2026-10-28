"""Launch 10M pure PPO after two complete solver consistency checks."""
from pathlib import Path
import os,json,subprocess,time
b=Path(__file__).resolve().parent;r=b/'contact_progress_20261028'
evidence=[]
for seed in [1062802,1062803]:
 p=r/f'forage_state_vel0_{seed}';s=json.loads((p/'state_audit.json').read_text());a=json.loads((p/'audit_results.json').read_text())
 if s['steps']<2399 or s['max_speed']>=5 or s['max_cache_error']>1e-4 or any(x['unfinished'] for x in a['rows']):raise RuntimeError('Preflight incomplete or extreme state: '+str(p))
 evidence.append(dict(path=str(p),**s))
out=b/'forage_repaired_long_20261028'
if out.exists():raise RuntimeError('Refuse to overwrite')
out.mkdir()
cmd=next(x['command'].copy() for x in json.loads((b/'scene_bottleneck_20261028/status.json').read_text()) if x['task']=='forage')
for flag,value in [('--output_dir',out),('--physics_batch_probe_envs',256),('--steps',10000000),('--eval_every',1048576),('--seed',2028),('--eval_seed',1062802)]:cmd[cmd.index(flag)+1]=str(value)
e=os.environ.copy()
for key in ['CONTACT_POLICY_AUDIT','CONTACT_AUDIT_MEAN_ONLY','CONTACT_AUDIT_MODE','CONTACT_DIAGNOSTIC','BRANCH_PLACEMENT_MANIFEST','PPO_PARENT_CHECKPOINT','GRAPH_TEACHER','BRANCH_COMPARISON_HAND','BRANCH_COMPARISON_MODE','BRANCH_COMPARISON_MANIFEST']:e.pop(key,None)
e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',CONTACT_PROGRESS_REWARD='1',CONTACT_PPO_GAMMA='.999',GRAPH_DRIVEN_WRIST='1',FORAGE_ZERO_VELOCITY_ITERATIONS='1',FORAGE_STATE_GUARD='1')
log=(b/'forage_repaired_long_20261028.log').open('x');p=subprocess.Popen(['timeout','14400']+cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,started_unix=time.time(),command=cmd,preflight=evidence,solver_repair='hand velocity iterations 8->0; leaf velocity iterations 4->0; position iterations unchanged',guard='abort entire run after 3 consecutive speed>10m/s with position finite difference<1m/s frames; no state clipping or changed success predicate',seed=2028,requested_steps=10000000,actual_steps=9994240),indent=2));print(p.pid)
