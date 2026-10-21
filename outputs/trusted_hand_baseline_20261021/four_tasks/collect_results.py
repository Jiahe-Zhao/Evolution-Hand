import json,math,hashlib
from pathlib import Path
base=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
roots=base/'four_task_results';rows={}
for d in roots.iterdir():
 if not (d/'report.json').exists():continue
 r=json.loads((d/'report.json').read_text());t=json.loads((d/'trace.json').read_text());r['first_step_speed_m_s']=math.sqrt(sum(v*v for v in t[0]['object'][7:10]));r['first_step_joint_excess_rad']=t[0]['max_joint_excess_rad'];r['stage_changes']=[{'time_s':x['time_s'],'stage':x['stage']} for i,x in enumerate(t) if i==0 or x['stage']!=t[i-1]['stage']];r['peak_speed_time_s']=max(t,key=lambda x:sum(v*v for v in x['object'][7:10]))['time_s'];r['initial_collision_audit']=json.loads((d/'startup_geometry.json').read_text()) if (d/'startup_geometry.json').exists() else None
 r['episode_digest_sha256']=hashlib.sha256((d/'trace.json').read_bytes()).hexdigest()
 if (d/'runtime_baseline.json').exists():
  a=json.loads((d/'runtime_baseline.json').read_text());r['model_sha256']=a['source_model_sha256'];r['actual_dt']=a['dt']
 if r['task']=='forage':
  r['max_leaf_contact_N']=[max(x['force_N'][i] for x in t) for i in (0,1)]
  if 'objects' in t[0]:r['leaf_final_xyz']={n:t[-1]['objects'][n][:3] for n in ['leaf_one','leaf_two']}
 rows[d.name]=r
(base/'four_task_review.json').write_text(json.dumps(rows,indent=2))
for n,r in rows.items():print(n,{k:r.get(k) for k in ['native_success','first_step_speed_m_s','peak_object_speed_m_s','max_joint_excess_rad','stage_changes']})
