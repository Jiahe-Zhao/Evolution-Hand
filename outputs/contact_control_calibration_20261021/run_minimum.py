"""Independent physics validation; never resumes training."""
import os,sys,json,subprocess,concurrent.futures
from pathlib import Path
base=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
cases=[('run15','nominal',['--record']),('run16','half_dt',['--dt',str(1/480)]),('run17','offset_minus_1mm',['--offset-mm','-1']),('run18','offset_plus_1mm',['--offset-mm','1']),('run19','joint_perturbation_0.01rad',['--joint-pulse','.01']),('run20','hand_gravity_on',['--hand-gravity'])]
common=['--headless','--plans',str(base/'plans_safe_audited.json'),'--duration','35','--kp','3','--kd','.3','--bounded-ik','--freeze-after-grasp','--contact-k','1500']
def run(case):
 name,label,extra=case
 command=[sys.executable,str(base/'calibrate_hand.py'),*common,'--output',str(base/name),*extra]
 with (base/(name+'.log')).open('w') as log:
  code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=900).returncode
 result={'run':name,'label':label,'exit_code':code,'report_exists':(base/name/'report.json').exists(),'command':command}
 print(json.dumps(result),flush=True);return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(run,cases):
  results.append(result);(base/'minimum_execution.json').write_text(json.dumps(results,indent=2))
