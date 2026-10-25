"""One bounded validation run; never resumes morphology evolution."""
import json,os,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent
command=[sys.executable,'-u',str(b/'worker.py'),'--generation','1','--handoff','standard','--mode','ppo','--learn-exploration','--num-envs','32','--smoke','--epochs','256','--critic-warmup','128','--ppo-lr','1e-7','--critic-lr','3e-4','--sigma-lr','1e-4','--entropy-coef','.001','--plans',str(b.parent/'contact_control_calibration_20261021/plans_wrap_cover_selected.json'),'--pad-face','ymixed','--kp','3','--kd','.3','--contact-k','1500','--hand-gravity','--episode-steps','1500','--eval-steps','1500','--push','--checkpoint',str(b.parent/'discrete_hand_evolution_20261024/seeds/g0/policy.pth'),'--seed','2027','--completion-reward','--gamma','.9995','--headless','--output',str(b/'run')]
s=dict(status='running',started=time.time(),command=command)
(b/'status.json').write_text(json.dumps(s,indent=2))
p=subprocess.Popen(command);s['pid']=p.pid;(b/'status.json').write_text(json.dumps(s,indent=2));code=p.wait()
s.update(status='completed' if code==0 and (b/'run/report.json').exists() else 'failed',exit_code=code,duration_s=time.time()-s['started'])
(b/'status.json').write_text(json.dumps(s,indent=2))
