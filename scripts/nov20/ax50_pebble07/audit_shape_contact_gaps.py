"""Offline rigid convex-distance checks of selected recorded frames; not a Newton replay."""
from pathlib import Path
import subprocess,json,sys
P=Path(__file__).parent;O=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07')
rows=[]
for tag,frames in [('shape_original_v1',[240,340,475]),('shape_topwide_v1',[190,220,240,300,475,525])]:
 for frame in frames:
  subprocess.run([sys.executable,str(P/'diagnose_gap.py'),tag,str(frame)],stdout=subprocess.DEVNULL,check=True)
  r=json.loads((O/f'{tag}_frame{frame}_gap_diagnostic.json').read_text());row=dict(tag=tag,frame=frame,time_s=r['time_s'])
  for finger,suffix in [('thumb','link_1_2'),('index','link_2_3')]:
   ps=[p for p in r['pairs'] if p['body'].endswith(suffix) and (p['contype'] or p['conaffinity'])]
   q=min(ps,key=lambda x:x['distance_mm']);row[finger+'_min_rigid_distance_mm']=q['distance_mm'];row[finger+'_geom']=q['geom']
  rows.append(row)
result=dict(scope='offline MuJoCo convex distance using actual recorded Newton body poses and per-run object XML; rigid geometry only, excludes deformable pad surfaces; not a replay of Newton contacts',rows=rows,assessment='topwide is not accepted: visually embedded in thumb, deep rigid overlap confirmed at hold frame; apparent lift is not a valid pinch')
(O/'shape_contact_geometry_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
