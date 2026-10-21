"""Run the wrapping grasp with the existing calibrated physics and drive settings."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--record',action='store_true');a=p.parse_args()
base=Path(__file__).resolve().parent
plans=base/'plans_wrap_cover_selected.json'
rows=json.loads(plans.read_text());assert rows and not rows[0]['intersections']
assert max([x['depth_mm'] for x in rows[0].get('self_intersections',[])]+[0])<.05
command=[sys.executable,str(base/'calibrate_wrap.py'),'--headless','--plans',str(plans),'--pad-face','ymixed','--duration','50','--kp','3','--kd','.3','--bounded-ik','--freeze-after-grasp','--contact-k','1500','--hand-gravity','--output',str(Path(a.output).resolve())]
if a.record:command.append('--record')
raise SystemExit(subprocess.call(command,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')))
