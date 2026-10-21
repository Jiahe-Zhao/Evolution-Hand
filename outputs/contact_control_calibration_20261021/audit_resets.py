import json,numpy as np
from pathlib import Path
from plan_grasp import Model
base=Path(__file__).resolve().parent
results=[]
for name in ['run15','run16','run17','run18','run19','run20']:
 if not (base/name/'audit.json').exists():continue
 r=json.loads((base/name/'audit.json').read_text());m=Model(r['args']['source'],face=r['args'].get('pad_face','x'));plan=r['plan'];rotation=np.array(plan['rotation']);center=np.array(plan['center'])+rotation@np.array([r['args']['offset_mm']/1000,0,0]);hits=m.collisions([plan['q'][n] for n in m.names],center,rotation)
 results.append({'run':name,'tool_intersections':hits,'max_depth_mm':max((h['depth_m']*1000 for h in hits),default=0.)})
print(json.dumps(results),flush=True);(base/'reset_audit.json').write_text(json.dumps(results,indent=2))
