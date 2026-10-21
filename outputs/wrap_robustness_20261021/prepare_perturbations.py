"""Predefined perturbations; reject intersecting resets before physics, report all."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
base=Path(__file__).resolve().parent;cal=base.parent/'contact_control_calibration_20261021';sys.path.insert(0,str(cal))
from plan_grasp import Model
source=base.parent/'motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf'
r=json.loads((cal/'plans_wrap_cover_selected.json').read_text())[0];m=Model(source,face='ymixed');q=[r['q'][n] for n in m.names];rotation=np.array(r['rotation'])
cases=[('nominal',0,0,0),('x_minus_05',-.5,0,0),('x_plus_05',.5,0,0),('y_minus_05',0,-.5,0),('y_plus_05',0,.5,0),('yaw_minus_2',0,0,-2),('yaw_plus_2',0,0,2),('x_minus_1',-1,0,0),('x_plus_1',1,0,0),('y_minus_1',0,-1,0),('y_plus_1',0,1,0),('mixed_minus',-.5,-.5,-1),('mixed_plus',.5,.5,1)]
rows=[]
for name,x,y,yaw in cases:
 c=np.array(r['center'])+rotation@np.array([x/1000,y/1000,0]);rot=rotation@Rotation.from_euler('z',yaw,degrees=True).as_matrix();hits=m.collisions(q,c,rot)
 depth=max([h['depth_m'] for h in hits]+[0]);rows.append(dict(name=name,x_mm=x,y_mm=y,yaw_deg=yaw,initial_intersections=hits,max_initial_penetration_mm=depth*1000,legal_reset=depth<=1e-6))
(base/'perturbation_cases.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
