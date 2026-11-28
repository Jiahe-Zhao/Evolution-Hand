import sys,json,numpy as np,xml.etree.ElementTree as ET
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Vt
from scipy.spatial.transform import Rotation
from mirror_mechanical_urdf import mirror
src,dst=map(Path,sys.argv[1:3]);dst.mkdir(parents=True,exist_ok=True)
mirror(src/'hand.urdf',dst/'hand.urdf')
# Keep all six wrist coordinates in the same mount convention.
tree=ET.parse(dst/'hand.urdf');base=ET.parse(src/'hand.urdf')
for j in tree.findall('joint'):
 if 'wrist_' in j.get('name'):
  a=base.find("joint[@name='%s']/axis"%j.get('name'));j.find('axis').set('xyz',a.get('xyz'))
tree.write(dst/'hand.urdf')
s=Usd.Stage.Open(str(src/'hand.usda'));s=Usd.Stage.Open(s.Flatten())
S=np.diag([-1.,1,1,1])
for p in s.Traverse():
 for a in p.GetAttributes():
  if not a.HasAuthoredValueOpinion():continue
  n=a.GetName();v=a.Get()
  if n in ('xformOp:translate','physics:centerOfMass','physics:localPos0','physics:localPos1'):
   a.Set(type(v)(-v[0],v[1],v[2]))
  if n=='xformOp:transform':a.Set(Gf.Matrix4d((S@np.array(v)@S).tolist()))
  if n in ('xformOp:orient','physics:principalAxes','physics:localRot0','physics:localRot1'):
   x,y,z=v.GetImaginary();q=Rotation.from_quat([x,-y,-z,v.GetReal()])
   if n.startswith('physics:localRot') and p.IsA(UsdPhysics.RevoluteJoint) and 'wrist_' not in p.GetName():
    axis=p.GetAttribute('physics:axis').Get()
    if axis in ('Y','Z'):q=q*Rotation.from_rotvec([np.pi,0,0])
   x,y,z,w=q.as_quat();a.Set(type(v)(w,type(v.GetImaginary())(x,y,z)))
  if n in ('points','normals'):
   arr=np.array(v);arr[:,0]*=-1;a.Set(Vt.Vec3fArray.FromNumpy(arr.astype('float32')))
  if n=='extent':
   arr=np.array(v);arr[:,0]=-arr[::-1,0].copy();a.Set(Vt.Vec3fArray.FromNumpy(arr.astype('float32')))
 if p.IsA(UsdGeom.Mesh):
  m=UsdGeom.Mesh(p);c=m.GetFaceVertexCountsAttr().Get();a=m.GetFaceVertexIndicesAttr();idx=np.array(a.Get());o=0
  for n in c:idx[o:o+n]=idx[o:o+n][::-1];o+=n
  a.Set(Vt.IntArray.FromNumpy(idx.astype('int32')))
s.GetRootLayer().Export(str(dst/'hand.usda'))
b=json.loads((src/'visual_binding.json').read_text())
for row in b['bindings']:
 a=np.array(row['bone_local_points']);a[:,:,0]*=-1;row['bone_local_points']=a.tolist()
(dst/'visual_binding.json').write_text(json.dumps(b))
