"""Locate inverted tetrahedra relative to mechanical collision meshes."""
import argparse,json
from pathlib import Path
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from audit_mechanical_workspace import graph,fk
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory)
c=json.loads((d/'palm_assembly_contract.json').read_text())
urdf=Path(next(v for v in c['files'] if v.endswith('.urdf')))
r=np.load(next(v for v in c['files'] if v.endswith('palm_tetrahedra.npz')))
f=np.load(d/'palm_failure.npz');pts=f['nodes'];tet=f['tetrahedra']
def edges(x):
 q=x[tet];return np.stack([q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]],axis=-1)
F=edges(pts)@np.linalg.inv(edges(r['points']));det=np.linalg.det(F);bad=np.where(det<=0)[0]
query=pts[tet[bad]].reshape(-1,3)
angles=dict(zip(f['joint_names'].tolist(),f['joint_positions'][0].tolist()))
nodes,base=graph(urdf);poses=fk(nodes,base,angles)
# Fixed Branch task root, quaternion wxyz=(.5,.5,.5,.5).
world=np.eye(4);world[:3,:3]=Rotation.from_quat([.5,.5,.5,.5]).as_matrix();world[:3,3]=[0,0,.4]
near=[]
for link in ET.parse(urdf).getroot().findall('link'):
 for ci,col in enumerate(link.findall('collision')):
  node=col.find('geometry/mesh')
  if node is None:continue
  mesh=trimesh.load(urdf.parent/node.get('filename'),force='mesh')
  mesh.apply_scale(np.fromstring(node.get('scale','1 1 1'),sep=' '))
  org=col.find('origin');T=np.eye(4)
  if org is not None:
   T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(org.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(org.get('xyz','0 0 0'),sep=' ')
  mesh.apply_transform(world@poses[link.get('name')]@T)
  _,dist,_=trimesh.proximity.closest_point_naive(mesh,query)
  near.append((float(dist.min()),link.get('name'),ci))
report={'inverted_tetrahedra':bad.tolist(),'minimum_volume_ratio':float(det.min()),'max_principal_stretch':float(np.linalg.svd(F,compute_uv=False).max()),'nearby_rigid_surfaces_m':sorted(near)[:12]}
(d/'failure_geometry_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
