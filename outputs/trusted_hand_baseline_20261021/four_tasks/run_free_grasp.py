import os,json,subprocess
from pathlib import Path
base=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021');d=json.load(open(base/'four_task_results/grasp_7/command.json'));cmd=d['args'];cmd[1]=str(base/'four_tasks/verify_free_grasp.py');out=base/'four_task_results/grasp_free_hold';out.mkdir(exist_ok=True);cmd[cmd.index('--output_dir')+1]=str(out)
env=dict(os.environ,**d['env'],BASELINE_RESET_SETTLE='1',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');(out/'command.json').write_text(json.dumps({'args':cmd,'env':d['env'],'reset_settle':True,'diagnostic_scene':'dynamic ball with gravity and removable support'},indent=2))
with (out/'process.log').open('w') as log:rc=subprocess.run(cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
print(rc)
