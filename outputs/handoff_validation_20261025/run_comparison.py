"""Small paired handoff checks; no morphology changes or additional BC."""
import json,os,subprocess,sys,time
from pathlib import Path
base=Path(__file__).resolve().parent
common=[sys.executable,str(base/'worker.py'),'--generation','1','--plans',str(base.parent/'contact_control_calibration_20261021/plans_wrap_cover_selected.json'),'--pad-face','ymixed','--kp','3','--kd','.3','--contact-k','1500','--hand-gravity','--num-envs','32','--smoke','--episode-steps','1500','--eval-steps','1500','--push','--checkpoint',str(base.parent/'discrete_hand_evolution_20261024/seeds/g0/policy.pth'),'--seed','2027','--completion-reward','--gamma','.9995','--headless']
status=[]
for label,options in [('control',['--handoff','control','--mode','eval','--critic-warmup','0']),('standard',['--handoff','standard','--mode','ppo','--epochs','256','--critic-warmup','128']),('calibrated',['--handoff','calibrated','--mode','ppo','--epochs','128','--critic-warmup','0'])]:
 out=base/label
 row=dict(name=label,status='running',start=time.time());status.append(row)
 (base/'status.json').write_text(json.dumps(status,indent=2))
 cmd=common+options+['--output',str(out)]
 with (base/(label+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid
  (base/'status.json').write_text(json.dumps(status,indent=2));code=p.wait()
 row.update(status='completed' if code==0 and (out/('control.json' if label=='control' else 'report.json')).exists() else 'failed',exit_code=code,duration=time.time()-row['start'])
 (base/'status.json').write_text(json.dumps(status,indent=2))
 if row['status']=='failed':break
