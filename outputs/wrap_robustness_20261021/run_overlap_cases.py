import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
base=Path(__file__).resolve().parent;cal=base.parent/'contact_control_calibration_20261021'
cases=[r for r in json.loads((base/'perturbation_cases.json').read_text()) if not r['legal_reset']];results=[]
def run(r):
 dest=base/('overlap_'+r['name'])
 cmd=[sys.executable,str(base/'perturb_grasp.py'),'--headless','--plans',str(cal/'plans_wrap_cover_selected.json'),'--pad-face','ymixed','--duration','50','--kp','3','--kd','.3','--bounded-ik','--freeze-after-grasp','--contact-k','1500','--hand-gravity','--offset-mm',str(r['x_mm']),'--offset-y-mm',str(r['y_mm']),'--yaw-deg',str(r['yaw_deg']),'--output',str(dest)]
 with (base/(dest.name+'.log')).open('w') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
 return dict(r,status='completed' if code==0 and (dest/'report.json').exists() else 'error',exit_code=code,command=cmd)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for r in pool.map(run,cases):
  results.append(r);(base/'overlap_execution.json').write_text(json.dumps(results,indent=2));print(r['name'],r['status'],flush=True)
