from pathlib import Path
import os,sys,json,subprocess
base=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
trials=[('strike_7_v2','strike_7_no_render',False,120,False),('strike_7_v2','strike_7_target_sync',True,120,True),('branch_7','branch_7_no_render',False,300,False),('forage_7_v2','forage_7_details',False,2400,False)]
for source,name,record,steps,sync in trials:
 data=json.load(open(base/'four_task_results'/source/'command.json'));cmd=data['args'];cmd[1]=str(base/'four_tasks/verify_diagnostic.py');cmd=[x for x in cmd if x!='--record_video'];cmd[cmd.index('--output_dir')+1]=str(base/'four_task_results'/name);cmd+=['--max_steps',str(steps)]
 if record:cmd+=['--record_video']
 out=base/'four_task_results'/name;out.mkdir(exist_ok=True)
 env=dict(os.environ,**data['env'],ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
 if sync:env['BASELINE_KINEMATIC_SYNC']='1'
 (out/'command.json').write_text(json.dumps({'args':cmd,'env':data['env'],'sync':sync},indent=2))
 with (out/'process.log').open('w') as log:rc=subprocess.run(cmd,cwd='/home/zjh/Evolution_PC/Isaaclab_other',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
 print(name,rc,flush=True)
