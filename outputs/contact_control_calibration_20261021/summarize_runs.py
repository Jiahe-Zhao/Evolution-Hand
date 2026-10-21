import json,sys
from pathlib import Path
base=Path(__file__).resolve().parent
summary=[]
for directory in sorted(base.glob('run[0-9][0-9]')):
 if not (directory/'report.json').exists():continue
 r=json.loads((directory/'report.json').read_text());trace=json.loads((directory/'trace.json').read_text());lt=r['lift_time_s']
 valid_seconds=0.;max_valid=0.
 for index,x in enumerate(trace):
  dt=trace[index]['t']-trace[index-1]['t'] if index else 1/60
  support=max(x.get('plate_force_N',0),x.get('support_force_N',0)) if 'support_force_N' in x else sum(v*v for v in x['plane_force'])**.5
  valid=x['rise_m']>.015 and x['relative_slip_m']<.005 and x['angle_rad']<.2 and support<.05
  valid_seconds=valid_seconds+dt if valid else 0.;max_valid=max(max_valid,valid_seconds)
 r['max_free_hold_s']=max_valid;r['passed_static_hold']=max_valid>=3.
 row={k:r[k] for k in ['reason','lift_time_s','max_free_hold_s','max_lift_m','peak_object_speed_m_s','max_limit_excess_rad']};row['run']=directory.name;row['valid_for_physics_conclusion']=directory.name not in ('run03','run04');row['excluded_reason']={'run03':'Initial joint state not explicitly written','run04':'Unmerged fixed-link collision semantics not yet corrected'}.get(directory.name);row['parameters']=r['run_parameters'];row['phases']={}
 for phase in ['close','lift','hold','lateral_force','torque','swing','transport','light_tap','retract']:
  samples=[x for x in trace if x['phase']==phase]
  if not samples:row['phases'][phase]={'status':'not_reached'};continue
  row['phases'][phase]={'time':[samples[0]['t'],samples[-1]['t']],'max_slip_mm':max(x['relative_slip_m'] for x in samples)*1000,'max_angle_deg':max(x['angle_rad'] for x in samples)*180/3.14159265,'max_normal_force_N':max(max(x['normal_force_N']) for x in samples),'last_normal_force_N':samples[-1]['normal_force_N']}
  if phase in ['lateral_force','torque','swing']:
   row['phases'][phase]['passed']=all(x['relative_slip_m']<.005 and x['angle_rad']<.2 and x['rise_m']>.015 and x.get('support_force_N',0)<.05 for x in samples)
 row['static_passed']=r['passed_static_hold'];row['complete_passed']=row['static_passed'] and all(row['phases'][p].get('passed',False) for p in ['lateral_force','torque','swing']) and any(x['phase']=='light_tap' and x.get('plate_force_N',0)>.1 for x in trace) and any(x['phase']=='retract' and x['rise_m']>.015 and x.get('plate_force_N',100)<.05 and x['relative_slip_m']<.005 for x in trace)
 summary.append(row)
 print(directory.name,'static',row['static_passed'],'hold',round(r['max_free_hold_s'],2),'lift_mm',round(r['max_lift_m']*1000,2),'forces',[row['phases'][p].get('passed') for p in ['lateral_force','torque','swing']],flush=True)
(base/'summary.json').write_text(json.dumps(summary,indent=2))
