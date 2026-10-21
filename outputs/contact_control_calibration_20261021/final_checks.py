import json,numpy as np
from pathlib import Path
base=Path(__file__).resolve().parent
results=[]
def strict_hold(x):
 return x['relative_slip_m']<.005 and x['angle_rad']<.2 and x['rise_m']>.015 and x.get('plate_force_N',100)<.05 and x.get('support_force_N',100)<.05
for name in ['run15','run16','run17','run18','run19','run20','run21']:
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
convergence={}
if all((base/name/'trace.json').exists() for name in ['run15','run16']):
 a=json.loads((base/'run15/trace.json').read_text());b=json.loads((base/'run16/trace.json').read_text());n=min(len(a),len(b));dp=np.array([np.linalg.norm(np.array(x['object'][:3])-np.array(y['object'][:3])) for x,y in zip(a[:n],b[:n])]);dq=np.array([np.max(abs(np.array(x['q'])-np.array(y['q']))) for x,y in zip(a[:n],b[:n])]);da=np.array([2*np.arccos(np.clip(abs(np.dot(x['object'][3:7],y['object'][3:7])),0,1)) for x,y in zip(a[:n],b[:n])]);convergence={'position_max_mm':float(dp.max()*1000),'position_rms_mm':float(np.sqrt(np.mean(dp**2))*1000),'joint_max_rad':float(dq.max()),'angle_max_deg':float(da.max()*180/np.pi),'criterion':'same time samples: position max <=5 mm, orientation max <=0.1 rad; both full scenarios pass','passed':bool(dp.max()<=.005 and da.max()<=.1 and all(r['passed'] for r in results if r['run'] in ['run15','run16']))}
 print('CONVERGENCE',convergence,flush=True)
(base/'minimum_results.json').write_text(json.dumps({'results':results,'timestep_convergence':convergence,'all_passed':all(any(x['run']==name and x['passed'] for x in results) for name in ['run15','run16','run17','run18','run19','run20']) and convergence.get('passed',False)},indent=2))
