"""Bounded diagnostics: fail if initialization/episode progress stops."""
import json,os,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;root=b/'scene_bottleneck_20261028';root.mkdir(exist_ok=False)
prior=json.loads((b/'contact_curriculum_20261028/status.json').read_text())
env=os.environ.copy();env.update(EVOLUTION_ROOT='/home/zjh/Evolution_PC',EVOLUTION_CODE_ROOT='/home/zjh/Evolution_PC',ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',BASELINE_RL_TOLERANCE='1',CONTACT_CURRICULUM='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0',STRIKE_TRAIN_STAGE='1')
rows=[]
def save():
 p=root/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(rows,indent=2));q.replace(p)
for task,n in [('forage',256),('branch',32),('strike',32)]:
 row0=next(r for r in prior if r['label']==task+'_check');cmd=row0['command'].copy();out=root/task
 cmd[cmd.index('--output_dir')+1]=str(out);cmd[cmd.index('--physics_batch_probe_envs')+1]=str(n)
 e=env.copy();e.update(CONTACT_DIAGNOSTIC='forage_surface' if task=='forage' else 'bottleneck',AUDIT_SOURCE_DIR=str(b/'contact_curriculum_20261028'/('strike_stage1' if task=='strike' else task)),GRAPH_DRIVEN_WRIST='1' if task!='branch' else '0',GRAPH_STRIKE_SUPPORT='1' if task=='strike' else '0')
 row=dict(task=task,status='running',started=time.time(),num_envs=n,command=cmd);rows.append(row)
 with (root/(task+'.log')).open('x') as f:
  p=subprocess.Popen(cmd,env=e,cwd='/home/zjh/Evolution_PC/Isaaclab_other',stdout=f,stderr=subprocess.STDOUT);row['pid']=p.pid;save();subprocess.run(['ionice','-c','2','-n','4','-p',str(p.pid)])
  while p.poll() is None:
   marker=out/'audit_progress.json';heartbeat=marker.stat().st_mtime if marker.exists() else row['started']
   if time.time()-heartbeat>240:
    row['failure_reason']='No simulation progress for 240 seconds';p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    break
   time.sleep(5)
 row.update(exit_code=p.returncode,status='completed' if p.returncode==0 and (out/'completed.json').exists() else 'failed',elapsed_s=time.time()-row['started']);save()
(root/'completed.json').write_text(json.dumps(rows,indent=2))
