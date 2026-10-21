from pathlib import Path
import os,sys,json,subprocess,shutil
root=Path('/home/zjh/Evolution_PC');work=root/'Isaaclab_other';base=root/'outputs/trusted_hand_baseline_20261021';scripts=base/'four_tasks'
for name in ['joint_graph_control.py','branch_reset_clearance_v5.py','imported_hand_geometry.py','strike_support_v5.py']:
 shutil.copy2(work/name,scripts/name)
lineage=work/'outputs/mechanical_evolution/exp_20261020_free_tool_v4_BC15g_v2/phenotypes/490138b7c5b1f0407c6d10edb72590f845a7be3958b40de7a0e449214d53964d/lineage.json'
rows=[]
for task in sys.argv[1:] or ['grasp','branch','forage','strike']:
 for seed in [7,8,9]:
  out=base/'four_task_results'/(f'{task}_{seed}'+os.environ.get('BASELINE_RUN_SUFFIX',''));out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,EVOLUTION_ROOT=str(root),EVOLUTION_CODE_ROOT=str(root),ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',GRAPH_SKIP_LOG_SCAN='1',EVOLUTION_STRIKE_SCENE_PRESET='open_grasp_v4',EVOLUTION_STRIKE_V4_TOOL_WIDTH_M='0.018',EVOLUTION_STRIKE_V4_PREGRASP='0.0')
  if task in ['forage','strike']:env['GRAPH_DRIVEN_WRIST']='1'
  if task=='strike':env['GRAPH_STRIKE_SUPPORT']='1'
  cmd=[sys.executable,str(scripts/'verify_four_tasks.py'),'--task',task,'--checkpoint',str(root/'outputs/strike_v3_training_20261020/bc_seeds'/task/'ppo/nn/mechanical_final.pth'),'--curriculum_stage','stage2','--legacy_source_urdf',str(work/'evaluation_results/preflight_grasp_projected_20260922/morphology/0_0/right/urdf/current_agent.urdf'),'--mechanical_asset_dir',str(base/'candidate'),'--lineage_json',str(lineage),'--individual_key','candidate','--headless','--output_dir',str(out),'--physics_batch_probe_envs','1','--seed',str(seed)]
  if seed==7:cmd+=['--record_video']
  (out/'command.json').write_text(json.dumps({'args':cmd,'env':{k:v for k,v in env.items() if k.startswith(('GRAPH_','EVOLUTION_'))}},indent=2))
  with (out/'process.log').open('w') as log:
   try:rc=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
   except subprocess.TimeoutExpired:rc=-999
  r=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else dict(task=task,seed=seed,error='missing report',exit_code=rc)
  rows.append(r);(base/'four_task_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(r),flush=True)
  if rc!=0:break
