
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from scipy.spatial.transform import Rotation
from itertools import product
rows={}
for name in ['original','topwide']:
 b=newton.ModelBuilder();b.add_mjcf(str(Path(__file__).parent/'assets'/f'object_{name}_v1.xml'),collapse_fixed_joints=False);m=b.finalize(device='cpu')
 T=m.shape_transform.numpy();S=m.shape_scale.numpy();B=m.body_q.numpy();points=[]
 for g in [0,1]:
  src=m.shape_source[g]
  if src is not None:
   v=np.asarray(src.vertices);f=np.asarray(src.indices).reshape(-1,3);volume=float(np.sum(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])))/6*1e9)
  else:v=np.array(list(product([-1,1],repeat=3)))
  v=Rotation.from_quat(T[g,3:]).apply(v*S[g])+T[g,:3]
  v=Rotation.from_quat(B[0,3:]).apply(v)+B[0,:3]
  points.extend(v)
 p=np.array(points);rows[name]=dict(newton_world_bounds_mm=(np.array([p.min(0),p.max(0)])*1000).tolist(),mesh_signed_volume_mm3=volume)
ref=json.loads((OUT/'topwide_compiled_geometry_audit.json').read_text())
for name in rows:
 rows[name]['max_bound_difference_from_mujoco_mm']=float(np.max(abs(np.array(rows[name]['newton_world_bounds_mm'])-np.array(ref[name]['assembly_world_bounds_mm']))))
 assert rows[name]['max_bound_difference_from_mujoco_mm']<1e-4
 assert rows[name]['mesh_signed_volume_mm3']>0
(OUT/'topwide_newton_geometry_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
