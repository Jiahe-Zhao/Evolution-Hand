"""Freeze a mechanical asset and replace its fixed palm filler by the FEM layer."""
import argparse
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    tree = ET.parse(source); root = tree.getroot()
    palm = root.find("link[@name='link_palm']")
    if palm is None:
        raise ValueError('Expected explicit fixed palm frame')
    removed = {'visuals': len(palm.findall('visual')), 'collisions': len(palm.findall('collision'))}
    for tag in ('visual', 'collision', 'inertial'):
        for node in palm.findall(tag):
            palm.remove(node)
    # A fixed coordinate frame is merged by the importer. Keep a tiny positive
    # inertial declaration for URDF compatibility, not the removed rigid filler.
    inertial = ET.SubElement(palm, 'inertial')
    ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value='1e-6')
    ET.SubElement(inertial, 'inertia', ixx='1e-9', iyy='1e-9', izz='1e-9', ixy='0', ixz='0', iyz='0')
    mesh_dir = destination.parent.parent/'meshes';mesh_dir.mkdir(parents=True,exist_ok=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    for mesh in root.findall('.//mesh'):
        path=(source.parent/mesh.get('filename')).resolve()
        shutil.copy2(path,mesh_dir/path.name)
        mesh.set('filename','../meshes/'+path.name)
    ET.indent(tree,space='  ');tree.write(destination,encoding='unicode',xml_declaration=True)
    destination.with_suffix('.compliant.json').write_text(json.dumps({
        'source':str(source),'replaced_fixed_palm_components':removed,
        'joint_mapping_unchanged':True,
        'requires_runtime_layer':'compliant_palm_runtime.spawn_compliant_palm',
        'status':'candidate_pending_dynamic_validation'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('destination')
    args=parser.parse_args();prepare(args.source,args.destination)
