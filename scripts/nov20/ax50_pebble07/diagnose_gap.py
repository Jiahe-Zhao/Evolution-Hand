"""Offline signed distance of original collision meshes at recorded body poses."""
from pathlib import Path
import re,json,sys,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation as R
from xml.etree.ElementTree import Element,SubElement,tostring
ROOT=Path('/home/zjh/Evolution_PC');O=ROOT/'outputs/nov20/ax50_pebble07'
tag=sys.argv[1] if len(sys.argv)>1 else 'pinch_trial_v4';a=np.load(O/(tag+'.npz'),allow_pickle=True);frame=int(sys.argv[2]) if len(sys.argv)>2 else -2;bq=a['body_q'][frame]
norm=lambda n:re.sub('[^a-zA-Z0-9]','',str(n))
lookup={norm(n):bq[i] for i,n in enumerate(a['body_names'])}
lookup.update({norm(str(n).split('/')[-1]):bq[i] for i,n in enumerate(a['body_names'])})
meta_path=O/(tag+'_result.json');meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
object_xml=Path(meta.get('object_xml',str(O/'object.xml')))
x=Element('mujoco');SubElement(x,'compiler',angle='radian');assets=SubElement(x,'asset');world=SubElement(x,'worldbody');labels=[]
for source,m in [('hand',mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'))),('object',mj.MjModel.from_xml_path(str(object_xml)))]:
 for g in range(m.ngeom):
  bn=m.body(int(m.geom_bodyid[g])).name;gn=m.geom(g).name or ''
  if source=='hand' and not any(bn.endswith(t) for t in ['link_1_2','link_2_3']):continue
  if source=='object' and gn not in ['stone0','foot0']:continue
  if norm(bn) not in lookup:raise ValueError(bn)
  p=lookup[norm(bn)];br=R.from_quat(p[3:]);gp=p[:3]+br.apply(m.geom_pos[g]);gr=br*R.from_quat(m.geom_quat[g][[1,2,3,0]])
  attrs=dict(name='g'+str(len(labels)),pos=' '.join(map(str,gp)),quat=' '.join(map(str,gr.as_quat()[[3,0,1,2]])))
  typ=m.geom_type[g]
  if typ==mj.mjtGeom.mjGEOM_MESH:
   k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];mn='m'+str(len(labels));SubElement(assets,'mesh',name=mn,vertex=' '.join(map(str,v.ravel())));attrs.update(type='mesh',mesh=mn)
  else:
   types={int(mj.mjtGeom.mjGEOM_SPHERE):'sphere',int(mj.mjtGeom.mjGEOM_CAPSULE):'capsule',int(mj.mjtGeom.mjGEOM_BOX):'box',int(mj.mjtGeom.mjGEOM_CYLINDER):'cylinder',int(mj.mjtGeom.mjGEOM_ELLIPSOID):'ellipsoid'}
   if int(typ) not in types:continue
   attrs.update(type=types[int(typ)],size=' '.join(map(str,m.geom_size[g])))
  SubElement(world,'geom',**attrs);labels.append(dict(source=source,body=bn,geom=gn,contype=int(m.geom_contype[g]),conaffinity=int(m.geom_conaffinity[g])))
mm=mj.MjModel.from_xml_string(tostring(x).decode());dd=mj.MjData(mm);mj.mj_forward(mm,dd);rows=[]
for i,l in enumerate(labels):
 if l['source']!='hand':continue
 for j,ll in enumerate(labels):
  if ll['source']!='object':continue
  seg=np.zeros(6);dist=mj.mj_geomDistance(mm,dd,i,j,.1,seg)
  rows.append(dict(**l,target=ll['geom'],distance_mm=float(dist*1000),closest_world_mm=(seg.reshape(2,3)*1000).tolist()))
rows.sort(key=lambda r:r['distance_mm']);out=dict(object_xml=str(object_xml),scope='offline MuJoCo convex collision distance at recorded Newton poses; rigid geometry only, not a Newton contact replay',time_s=float(a['time'][frame]),pairs=rows)
(O/(tag+('_frame'+str(frame) if len(sys.argv)>2 else '')+'_gap_diagnostic.json')).write_text(json.dumps(out,indent=2));print(json.dumps(rows[:12],indent=2))
