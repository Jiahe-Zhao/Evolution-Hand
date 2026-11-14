from pathlib import Path
import json,sys,re,xml.etree.ElementTree as ET
from pxr import Usd,UsdGeom,UsdPhysics,Gf
import numpy as np
src,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True,parents=True);k=1.1
thumb={f'link_1_{i}' for i in range(4)}
stage=Usd.Stage.Open(str(src/'hand.usda'));stage=Usd.Stage.Open(stage.Flatten())
def mul(p,key,f):
 a=p.GetAttribute(key)
 if a and a.HasAuthoredValueOpinion():a.Set(a.Get()*f)
for p in stage.Traverse():
 name=p.GetName()
 if name in thumb:
  for key,f in [('physics:mass',k**3),('physics:centerOfMass',k),('physics:diagonalInertia',k**5)]:mul(p,key,f)
 if p.IsA(UsdPhysics.Joint):
  for i in [0,1]:
   path=p.GetRelationship(f'physics:body{i}').GetTargets()
   if path and path[0].name in thumb:mul(p,f'physics:localPos{i}',k)
 if any(x in thumb for x in str(p.GetParent().GetPath()).split('/')):
  mul(p,'xformOp:translate',k)
  if p.IsA(UsdGeom.Gprim):
   xf=UsdGeom.Xformable(p);xf.AddScaleOp(opSuffix='thumb_length').Set(Gf.Vec3f(k))
stage.GetRootLayer().Export(str(out/'hand.usda'))
tree=ET.parse(src/'hand.urdf')
for l in tree.findall('link'):
 if l.get('name') not in thumb:continue
 for o in l.findall('.//origin'):o.set('xyz',' '.join(str(k*float(x)) for x in o.get('xyz','0 0 0').split()))
 for m in l.findall('.//mesh'):m.set('scale',' '.join(str(k*float(x)) for x in m.get('scale','1 1 1').split()))
 for m in l.findall('inertial/mass'):m.set('value',str(k**3*float(m.get('value'))))
 for m in l.findall('inertial/inertia'):
  for a in m.attrib:m.set(a,str(float(m.get(a))*k**5))
for j in tree.findall('joint'):
 if j.find('parent').get('link') in thumb:
  o=j.find('origin');o.set('xyz',' '.join(str(k*float(x)) for x in o.get('xyz').split()))
tree.write(out/'hand.urdf',encoding='utf-8',xml_declaration=True)
binding=json.loads((src/'visual_binding.json').read_text())
for b in binding['bindings']:
 v=np.array(b['bone_local_points'])
 for i,bone in enumerate(b['bones']):
  if bone in thumb:v[:,i]*=k
 b['bone_local_points']=v.tolist()
(out/'visual_binding.json').write_text(json.dumps(binding))
