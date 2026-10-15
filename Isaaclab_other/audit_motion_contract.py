"""Export and verify the 20-DOF mechanical hand motion contract."""
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('assets');p.add_argument('legacy');a=p.parse_args()
root=Path(a.assets).resolve();legacy=Path(a.legacy).resolve()
paths={s:root/s/'urdf/current_agent.urdf' for s in ('right','left')}
xml={s:ET.parse(v).getroot() for s,v in paths.items()}
joints={s:{j.get('name'):j for j in x.findall('joint') if j.get('type')!='fixed'} for s,x in xml.items()}
assert set(joints['right'])==set(joints['left']) and len(joints['right'])==20
source={j.get('name'):j for j in ET.parse(legacy).getroot().findall('joint') if j.get('type')!='fixed'}
assert set(source)==set(joints['right'])
reflection=np.diag([-1.,1.,1.]);rows=[]
for name,r in joints['right'].items():
 l=joints['left'][name];sr=source[name]
 ar=np.fromstring(r.find('axis').get('xyz'),sep=' ');al=np.fromstring(l.find('axis').get('xyz'),sep=' ')
 assert np.allclose(al,-reflection@ar,atol=1e-8),name
 lim={k:float(r.find('limit').get(k)) for k in ('lower','upper','effort','velocity')}
 for k in lim:assert np.isclose(lim[k],float(l.find('limit').get(k))),name
 assert lim['lower']<=0<=lim['upper'],name
 origins={}
 for side,j in [('right',r),('left',l)]:
  o=j.find('origin');origins[side]=(np.fromstring(o.get('xyz','0 0 0'),sep=' '),Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix())
 assert np.allclose(origins['left'][0],reflection@origins['right'][0],atol=1e-8),name
 assert np.allclose(origins['left'][1],reflection@origins['right'][1]@reflection,atol=1e-8),name
 rows.append(dict(name=name,zero_rad=0.,right_axis=ar.tolist(),left_axis=al.tolist(),limits=lim,legacy_limits={k:float(sr.find('limit').get(k)) for k in ('lower','upper')}))
files={str(legacy)}
for side,path in paths.items():
 files.add(str(path))
 for m in xml[side].findall('.//mesh'):files.add(str((path.parent/m.get('filename')).resolve()))
for sub in ('compliant_palm','compliant_palm_left'):
 for name in ('compliant_palm.json','palm_tetrahedra.npz','continuous_palm.stl'):files.add(str(root/sub/name))
report={'physical_dofs':20,'mirror_contract_verified':True,'zero_pose':'All joint coordinates zero','joint_order_note':'Rows are URDF order; action mapping must use controller/runtime joint names','thumb_transfer':'Legacy negative excursion mapped to positive normalized excursion; not Cartesian trajectory equivalence','joints':rows,'asset_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sorted(files)}}
(root/'motion_contract.json').write_text(json.dumps(report,indent=2));print(json.dumps({'physical_dofs':20,'mirror_contract_verified':True,'hashed_files':len(files),'output':str(root/'motion_contract.json')}))
