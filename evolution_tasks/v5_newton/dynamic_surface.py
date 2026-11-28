"""Visual linear-blend skinning. Does not write physics or contact geometry."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
class DynamicSurface:
 def __init__(self,path):
  self.data=json.loads(Path(path).read_text())
  self.bindings={x['path']:x for x in self.data['bindings']}
 def world_points(self,path,poses):
  b=self.bindings[path];local=np.asarray(b['bone_local_points']);weights=np.asarray(b['weights']);parts=[]
  for i,name in enumerate(b['bones']):
   p=np.asarray(poses[name]);parts.append(Rotation.from_quat(p[3:]).apply(local[:,i])+p[:3])
  return (np.stack(parts,1)*weights[:,:,None]).sum(1)
 def update_stage(self,stage,poses):
  """Call immediately before rendering after physics has updated body poses."""
  from pxr import UsdGeom,Gf
  for path,b in self.bindings.items():
   parent=np.asarray(poses[b['parent']]);points=Rotation.from_quat(parent[3:]).inv().apply(self.world_points(path,poses)-parent[:3])
   mesh=UsdGeom.Mesh.Get(stage,path);mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in points])
   # A deformed surface cannot keep undeformed vertex normals.
   mesh.GetNormalsAttr().Clear()
