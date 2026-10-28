from pathlib import Path
import os,json,subprocess
b=Path(__file__).resolve().parent
for seed in [1062803,1062804]:
 out=b/'forage_evaluation_videos_20261028'/str(seed);out.mkdir(parents=True,exist_ok=True)
 cmd=json.loads((b/'forage_repaired_long_20261028/launch.json').read_text())['command'].copy()
 for flag,val in [('--output_dir',out),('--physics_batch_probe_envs',1)]:cmd[cmd.index(flag)+1]=str(val)
 cmd+=['--record_video'];e=os.environ.copy();e.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',CONTACT_PROGRESS_REWARD='1',CONTACT_PPO_GAMMA='.999',GRAPH_DRIVEN_WRIST='1',FORAGE_ZERO_VELOCITY_ITERATIONS='1',FORAGE_STATE_GUARD='1',FORAGE_POLICY_VIDEO=str(b/'forage_repaired_long_20261028/final.pth'),FORAGE_VIDEO_ACTION_SEED=str(seed))
 with (out/'worker.log').open('w') as f:r=subprocess.run(['timeout','900']+cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=e,stdout=f,stderr=subprocess.STDOUT)
 (out/'launch.json').write_text(json.dumps(dict(command=cmd,action_seed=seed,returncode=r.returncode),indent=2))
