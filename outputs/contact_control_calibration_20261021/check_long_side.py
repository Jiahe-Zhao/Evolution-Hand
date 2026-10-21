import json,numpy as np
from pathlib import Path
base=Path(__file__).resolve().parent
results=[]
def strict_hold(x):
 return x['relative_slip_m']<.005 and x['angle_rad']<.2 and x['rise_m']>.015 and x.get('plate_force_N',100)<.05 and x.get('support_force_N',100)<.05
import sys
for name in sys.argv[1:]:
 folder=base/name
 if not (folder/'report.json').exists():continue
 r=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'trace.json').read_text());lt=r['lift_time_s'];checks={}
 checks['free_hold_3s']=r['passed_static_hold'];checks['finite_and_limits']=all(np.isfinite(x['q']).all() and np.isfinite(x['object']).all() for x in rows) and r['max_limit_excess_rad']<.03
 for phase,duration in [('lateral_force',.95),('torque',.95),('swing',2.9),('transport',4.9)]:
  chunk=[x for x in rows if x['phase']==phase];checks[phase]=bool(chunk) and chunk[-1]['t']-chunk[0]['t']>=duration and all(strict_hold(x) for x in chunk)
 tap=[x for x in rows if x['phase']=='light_tap'];retract=[x for x in rows if x['phase']=='retract'];tail=[x for x in retract if x['t']>rows[-1]['t']-1]
 checks['light_contact']=any(x.get('plate_force_N',0)>.1 for x in tap)
 checks['retained_after_contact']=bool(tail) and tail[-1]['t']-tail[0]['t']>.9 and all(strict_hold(x) and x['normal_force_N'][0]>.1 and sum(x['normal_force_N'][1:])>.1 for x in tail)
 row={'run':name,'passed':all(checks.values()),'checks':checks,'lift_time_s':lt,'max_free_hold_s':r['max_free_hold_s'],'peak_plate_force_N':max(x.get('plate_force_N',0) for x in rows),'peak_limit_excess_rad':r['max_limit_excess_rad'],'final_slip_mm':rows[-1]['relative_slip_m']*1000,'final_angle_deg':rows[-1]['angle_rad']*180/np.pi,'phases':sorted(set(x['phase'] for x in rows))}
 results.append(row);print(name,row['passed'],checks,flush=True)

(base/'long_side_checks.json').write_text(json.dumps(results,indent=2))
