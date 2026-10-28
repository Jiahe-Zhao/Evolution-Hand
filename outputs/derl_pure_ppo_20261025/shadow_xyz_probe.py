"""Inspect the stock Shadow asset before composing an independent XYZ wrapper."""
import os,json
from pathlib import Path
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
from pxr import Usd,UsdPhysics
b=Path(__file__).resolve().parent
source=json.loads((b/'branch_long_pair_20261028/shadow/asset_descriptor.json').read_text())['robot_cfg']['spawn']['usd_path']
s=Usd.Stage.Open(source)
if s is None:raise RuntimeError('Shadow asset unavailable')
rows=[]
for p in s.Traverse():
 if p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.Joint):
  rows.append(dict(path=str(p.GetPath()),type=p.GetTypeName(),articulation_root=p.HasAPI(UsdPhysics.ArticulationRootAPI),body0=[str(x) for x in p.GetRelationship('physics:body0').GetTargets()],body1=[str(x) for x in p.GetRelationship('physics:body1').GetTargets()]))
out=b/'shadow_xyz_20261029';out.mkdir(exist_ok=True)
(out/'stock_inventory.json').write_text(json.dumps(dict(source=source,default_prim=str(s.GetDefaultPrim().GetPath()),rows=rows),indent=2))
s.Flatten().Export(str(out/'stock_flattened.usda'))
app.close()
