"""Audit joint axis redundancy and pad-normal actuation at the planned posture."""
import sys,json,numpy as np
from pathlib import Path
from plan_centered_pad_faces import Model
base=Path(__file__).parents[1];m=Model(base/'candidate/right/urdf/current_agent.urdf');r=json.loads((base/'centered_plans_inward.json').read_text())[0];q=np.array([r['q'][n] for n in m.names]);p,n=m.pads(q);rows=[]
for f in range(1,6):
 ids=[i for i,name in enumerate(m.names) if name.startswith(f'link_{f}_') or f'to_link_{f}_' in name];J=[]
 for i in ids:
  h=q.copy();h[i]+=1e-6;pp,_=m.pads(h);J.append((pp[f-1]-p[f-1])/1e-6)
 J=np.array(J).T;rows.append({'finger':f,'joints':[m.names[i] for i in ids],'singular_values_m_per_rad':np.linalg.svd(J,compute_uv=False).tolist(),'pad_normal_jacobian_norm_m_per_rad':float(np.linalg.norm(n[f-1]@J))})
(base/'opposition_axes_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
