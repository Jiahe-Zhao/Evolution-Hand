
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
root=Path((str(_EVOLUTION_ROOT)));out=root/'outputs/nov20/ax50_handle13';out.mkdir(exist_ok=True);source=root/'outputs/nov20/handle.xml';tree=_evo_parse_xml(source);r=tree.getroot();w=r.find('worldbody');w.remove(w.find("body[@name='diagnostic_world']"))
for tag in ['actuator','asset','contact','sensor','keyframe','tendon']:
 node=r.find(tag)
 if node is not None:r.remove(node)
clean=out/'object.xml';tree.write(clean)
if '--ax50' in sys.argv:
 sys.path.insert(0,str(root/'outputs/ax50_visualskin_v3_0_0'));import ax50_visualskin
import newton
b=newton.ModelBuilder();b.add_mjcf(str(clean));model=b.finalize(device='cpu');data=dict(scope='asset import only, no hand operation validation',source=str(source),newton_module=newton.__file__,body_names=model.body_label,mass_kg=model.body_mass.numpy().tolist(),dof_count=model.joint_dof_count,joint_names=model.joint_label,shape_count=model.shape_count,authored_geoms=[dict(x.attrib) for x in w.iter('geom')]);tag='asset_ax50' if '--ax50' in sys.argv else 'asset_old';(out/(tag+'.json')).write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='authored_geoms'},indent=2))
