"""Bounded sequential verification; never start/resume evolution or PPO."""
import os,sys,subprocess,json
from pathlib import Path
base=Path(__file__).resolve().parent
env=dict(os.environ,ISAACLAB_ROOT='/home/zjh/IsaacLab',CONDA_PREFIX='/home/zjh/miniconda3/envs/evolution_isaaclab',ACCEPT_EULA='Y',PRIVACY_CONSENT='Y',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
tests=[('baseline_final','baseline','right',['--record','--full-range']),
       ('added','add_link_link_2_3','right',[]),('removed','remove_link_link_2_1','right',[]),
       ('left','baseline','left',[]),('half_dt','baseline','right',['--dt',str(1/480),'--full-range'])]
records=[]
for label,variant,side,extra in tests:
    out=base/'runtime'/label
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cmd=[sys.executable,str(base/'test_motion.py'),'--urdf',str(base/'matrix'/variant/side/'urdf/current_agent.urdf'),'--output',str(out),'--headless','--device','cuda:0',*extra]
    with (out/'process.log').open('w') as log:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
    report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
    records.append({'label':label,'exit_code':code,'passed':report.get('passed',False),'report':str(out/'report.json')})
    (base/'runtime_summary.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
bench=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
for mode in ['press','hold']:
    out=base/'runtime'/('pad_'+mode);out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,str(bench/'test_baseline.py'),'--source',str(base/'matrix/baseline/right/urdf/current_agent.urdf'),'--output',str(out),'--mode',mode,'--headless','--device','cpu']
    with (out/'process.log').open('w') as log:code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
    report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else {}
    records.append({'label':'pad_'+mode,'exit_code':code,'passed':report.get('passed',False),'report':str(out/'report.json')})
    (base/'runtime_summary.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
