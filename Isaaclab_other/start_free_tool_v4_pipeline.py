"""Verify demonstrations, initialize BC, probe 4096 environments, then launch v4."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,time
root=Path('/home/zjh/Evolution_PC');work=root/'Isaaclab_other'
out=root/'outputs/strike_v3_training_20261020'
assets=work/'outputs/mechanical_evolution/rigid_palm_smoke_20261017_baseline/phenotypes/a1104f889e1574a32bee519c96e7d5490250a9ecec9aa54edd517575b472d272/assets'
env=dict(os.environ,EVOLUTION_ROOT=str(root),ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',TERM='xterm',PYTHONUNBUFFERED='1',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',CUDA_VISIBLE_DEVICES='0',ISAAC_SIM_SETUP='/dev/null',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',EVOLUTION_STRIKE_BC_MODE='1',EVOLUTION_BRANCH_BC_MODE='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0')
status={'experiment':'exp_20261020_free_tool_v4_BC15g_v2','stages':[]}
def save(): (out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
def run(label,cmd,timeout=3600):
    print('START',label,flush=True);status['current_stage']=label;save()
    start=time.monotonic()
    with (out/(label+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    status['stages'].append(dict(name=label,returncode=result.returncode,elapsed_seconds=round(time.monotonic()-start,2)));save()
    if result.returncode:raise RuntimeError(f'{label} failed: {out/(label+".log")}')
    print('DONE',label,flush=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest=out/'final_suite/manifest.json'
run('verify_scripts',[sys.executable,str(work/'validate_mechanical_v4_suite.py'),'--assets',str(assets),'--output',str(manifest.parent),'--seeds','7','--strike-seeds','8','9'])
# Seed 7 already passed in the identical scene; retain it without replaying it.
data=json.loads(manifest.read_text());base=out/'vertical_stroke_script'
entry={'metrics':str(base/'metrics.json'),'trace':str(base/'metrics.trace.npz'),'scene':str(base/'mechanical_scene_contract.json'),'seed':7}
for field in ('metrics','trace','scene'):entry[field+'_sha256']=sha(entry[field])
assert json.loads(Path(entry['metrics']).read_text())['success']
a=json.loads(Path(entry['scene']).read_text());b=json.loads(Path(data['tasks']['strike'][0]['scene']).read_text())
for field in ('version','effective_settings','implementation_sha256','mechanical_urdf_sha256','curriculum_stage'):assert a[field]==b[field],field
assert all(json.loads(Path(item['metrics']).read_text())['success'] for item in data['tasks']['strike'])
data['tasks']['strike'].insert(0,entry);manifest.write_text(json.dumps(data,indent=2))
run('initialize_bc',[sys.executable,str(work/'initialize_mechanical_bc.py'),'--assets',str(assets),'--manifest',str(manifest),'--output',str(out/'bc_seeds')])
seeds=json.loads((out/'bc_seeds/seeds.json').read_text())
checkpoint=seeds['tasks']['strike']['checkpoint']
common=[sys.executable,str(work/'evaluate_mechanical_task.py'),'--task','strike','--checkpoint',checkpoint,'--curriculum_stage','stage2','--legacy_source_urdf',str(work/'evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf'),'--mechanical_asset_dir',str(assets),'--lineage_json',str(assets.parent/'lineage.json'),'--individual_key','candidate','--headless']
run('batch4096',common+['--output_dir',str(out/'batch4096'),'--physics_batch_probe_envs','4096','--physics_batch_probe_zero_actions'],timeout=1200)
probe=json.loads((out/'batch4096/physics_batch_probe.json').read_text());assert probe['passed'] and probe['num_envs']==4096
run('bc_strike_evaluation',common+['--output_dir',str(out/'bc_strike_evaluation'),'--seed','7'],timeout=600)
# Autonomous BC success is measured, not required: PPO is the next learning stage.
status['bc_strike_evaluation']=json.loads((out/'bc_strike_evaluation/evaluation.json').read_text())['success_rate']
lineage=work/(status['experiment']+'.json')
assert not lineage.exists(), 'Experiment already exists; do not duplicate or overwrite it'
log_path=Path('/tmp/formal_free_tool_v4_20261020.log')
with log_path.open('w') as log:
    process=subprocess.Popen(['bash',str(work/'run_formal_free_tool_v4_20261020.sh')],cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
status.update(training_pid=process.pid,training_log=str(log_path),current_stage='training_started');save()
print('TRAINING_STARTED',process.pid,str(log_path),flush=True)
video=out/'video';video.mkdir(exist_ok=True)
run('record_strike_video',[sys.executable,str(root/'evolution_tasks/task_suite/mechanical_strike_v4_demo.py'),'strike','--mechanical_urdf',str(assets/'right/urdf/current_agent.urdf'),'--strike_policy_demo','--strike_contact_closure','--strike_vertical_stroke','--strike_post_grasp_dwell_steps','40','--strike_max_steps','1200','--output',str(video/'strike_free_tool_v4.mp4'),'--metrics',str(video/'metrics.json'),'--seed','7','--training_scene','--headless','--device','cuda:0'],timeout=1200)
assert json.loads((video/'metrics.json').read_text())['success'], 'Recorded demonstration failed'
status['current_stage']='training_started_video_verified';save();print('VIDEO_VERIFIED',flush=True)
