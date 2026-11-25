from pathlib import Path
import json,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from collections import Counter
R=Path('/home/zjh/Evolution_PC');P=R/'scripts/nov20/ax50_pebble07';O=R/'outputs/nov20/ax50_pebble07'
from xml.etree import ElementTree as E
out={}
for label in ['original','topwide']:
 path=P/'assets'/f'object_{label}_v1.xml';m=mj.MjModel.from_xml_path(str(path));i=m.body('item').id
 points=[]
 for g in range(m.ngeom):
  if m.geom_bodyid[g]!=i:continue
  if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
   k=m.geom_dataid[g];v=np.array(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]])
  else:
   assert m.geom_type[g]==mj.mjtGeom.mjGEOM_BOX
   from itertools import product
   v=np.array(list(product([-1,1],repeat=3)))*m.geom_size[g]
  q=m.geom_quat[g];rr=Rotation.from_quat(q[[1,2,3,0]])
  points.extend(rr.apply(v)+m.geom_pos[g])
 points=np.array(points);q=m.body_quat[i];world=Rotation.from_quat(q[[1,2,3,0]]).apply(points)+m.body_pos[i]
 mesh=E.parse(path).find("asset/mesh[@name='peb0']")
 v=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3);f=np.fromstring(mesh.get('face'),sep=' ',dtype=int).reshape(-1,3) if mesh.get('face') else ConvexHull(v).simplices
 edges=Counter(tuple(sorted((int(a),int(b)))) for face in f for a,b in zip(face,np.roll(face,-1)))
 area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
 out[label]=dict(mass_g=float(m.body_mass[i]*1000),assembly_local_bounds_mm=(np.array([points.min(0),points.max(0)])*1000).tolist(),assembly_world_bounds_mm=(np.array([world.min(0),world.max(0)])*1000).tolist(),all_edges_two_faces=all(n==2 for n in edges.values()),min_triangle_area_mm2=float(min(area)*1e6),convex_hull_volume_mm3=float(ConvexHull(v).volume*1e9))
for key in ['assembly_local_bounds_mm','assembly_world_bounds_mm']:
 out[key+'_max_difference']=float(np.max(np.abs(np.array(out['original'][key])-np.array(out['topwide'][key]))))
 assert out[key+'_max_difference']<1e-5
assert out['original']['mass_g']==out['topwide']['mass_g']
assert out['topwide']['all_edges_two_faces'] and out['topwide']['min_triangle_area_mm2']>0
(O/'topwide_compiled_geometry_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
