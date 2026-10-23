"""Compile a ternary genotype using the calibrated hand compiler, then audit it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from genes import decode, validate


def mesh_of(node, urdf):
    import numpy as np
    import trimesh
    from mechanical_contact_model import pose
    ref = node.find('geometry/mesh')
    mesh = trimesh.load(urdf.parent/ref.get('filename'), force='mesh')
    mesh.apply_scale(np.fromstring(ref.get('scale', '1 1 1'), sep=' '))
    mesh.apply_transform(pose(node.find('origin')))
    return mesh


def adjust_mechanical(path, baseline_width, metadata):
    import numpy as np
    from mechanical_contact_model import inertia
    tree = ET.parse(path); root = tree.getroot()
    code = metadata['code']
    rows = []
    for link in root.findall('link'):
        palm = link.get('name') == 'link_palm_chassis'
        pad = link.find("collision[@name='finger_pad']")
        if not ((palm and code[0]) or (pad is not None and code[7])):
            continue
        tag = 'rigid_palm_chassis' if palm else 'finger_pad'
        node = link.find(f"collision[@name='{tag}']")
        body = mesh_of(node, path)
        center = body.bounds.mean(0)
        scale = np.ones(3)
        # Pad width is its local X dimension, independent of bearing diameter.
        scale[2 if palm else 0] = metadata['palm_width_m']/body.extents[2] if palm else metadata['pad_width_scale']
        body.vertices = (body.vertices-center)*scale+center
        pieces = [(body, 1100.)]
        if not palm:
            for bearing in link.findall('visual'):
                if bearing.get('name') not in ('proximal_bearing', 'distal_bearing'):
                    continue
                metal = mesh_of(bearing, path)
                # Check containment against the actual convex surface, including
                # rounded corners; never silently shrink a bearing or clip a gene.
                normals = body.face_normals
                points = body.triangles_center
                margin = -(metal.vertices[:, None, :]-points[None, :, :])*normals[None, :, :]
                clearance = float(margin.sum(2).min())
                if clearance < .0001:
                    raise ValueError(f"{link.get('name')}: pad level exposes bearing (clearance {clearance})")
                pieces.append((metal, 1600.))
        # Replace geometry shared by visual and collision; inertia includes metal.
        target = path.parent/node.find('geometry/mesh').get('filename')
        body.export(target)
        mass = inertia(link, pieces)
        rows.append(dict(link=link.get('name'), dimensions_m=body.extents.tolist(), mass_kg=mass))
    ET.indent(tree, space='  ')
    if rows:
        tree.write(path, encoding='unicode', xml_declaration=True)
    return rows


def build(baseline_path, code, output, compiler_dir):
    import numpy as np
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(Path(compiler_dir).resolve()))
    from code_to_urdf import generate_urdf_from_dict
    from compile_mechanical_hand import compile_hand
    from mirror_mechanical_urdf import mirror
    from collision_gate import audit_generated_urdf
    from mechanical_contact_model import inertia
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(Path(baseline_path).read_text())
    # Measure the actual generated baseline chassis, not the root spread alone.
    base_bone = output/'reference/bone/urdf/current_agent.urdf'
    generate_urdf_from_dict(baseline, str(base_bone.parent.parent/'meshes'), str(base_bone))
    base_mech = output/'reference/right/urdf/current_agent.urdf'
    base_mapping = compile_hand(base_bone, base_mech)
    width = base_mapping['rigid_palm']['dimensions_m'][2]
    hand, metadata = decode(baseline, code, width)
    bone = output/'bone/urdf/current_agent.urdf'
    generate_urdf_from_dict(hand, str(bone.parent.parent/'meshes'), str(bone))
    if code[6]:
        tree = ET.parse(bone); joint = tree.getroot().find("joint[@name='link_1_thumb_spread_joint']")
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        axis = Rotation.from_rotvec([0, 0, metadata['thumb_axis_rotation_rad']]).apply(axis)
        joint.find('axis').set('xyz', ' '.join(map(str, axis)))
        tree.write(bone, encoding='unicode', xml_declaration=True)
    right = output/'right/urdf/current_agent.urdf'
    mapping = compile_hand(bone, right)
    changed = adjust_mechanical(right, width, metadata)
    # Bone terminal capsule width is changed independently, after mechanical
    # compilation so mechanical bearing size is never changed by the pad gene.
    if code[7]:
        tree = ET.parse(bone)
        for link in tree.getroot().findall('link'):
            if link.get('name') not in {f'link_{d}_3' for d in range(1, 6)}:
                continue
            seen = set()
            for node in link.findall('visual')+link.findall('collision'):
                ref = node.find('geometry/mesh'); file = (bone.parent/ref.get('filename')).resolve()
                if file in seen: continue
                seen.add(file)
                import trimesh
                mesh = trimesh.load(file, force='mesh'); mesh.apply_scale([metadata['pad_width_scale'], 1, 1]); mesh.export(file)
            density = next(x.get('density', 850.) for x in hand['links'] if x['name_code'] == link.get('name'))
            inertia(link, [(mesh_of(link.find('visual'), bone), density)])
        tree.write(bone, encoding='unicode', xml_declaration=True)
    left = output/'left/urdf/current_agent.urdf'; mirror(right, left)
    checks = {}
    for label, path in [('bone', bone), ('right', right), ('left', left)]:
        passed, audit = audit_generated_urdf(path)
        checks[label] = dict(passed=bool(passed), reasons=audit.get('reasons', []))
    metadata.update(changed_parts=changed, checks=checks, passed=all(v['passed'] for v in checks.values()),
                    compiler_sha256={name:hashlib.sha256((Path(compiler_dir)/name).read_bytes()).hexdigest()
                                     for name in ['code_to_urdf.py','compile_mechanical_hand.py','mechanical_contact_model.py','mirror_mechanical_urdf.py']},
                    artifacts={k:str(v) for k,v in [('bone',bone),('right',right),('left',left)]},
                    baseline_width_m=width, joint_count=mapping['actuated_joint_count'],
                    geometry_only=True, dynamic_validation=False)
    (output/'genotype.json').write_text(json.dumps(hand, indent=2))
    (output/'candidate.json').write_text(json.dumps(metadata, indent=2))
    # Keep the ordinary mapping manifest accurate after the pad/palm updates.
    mapping['discrete_genes'] = metadata
    mapping['urdf_sha256'] = hashlib.sha256(right.read_bytes()).hexdigest()
    physical = {l.get('name'):l for l in ET.parse(right).getroot().findall('link')}
    mapping['mass_kg'] = sum(float(l.find('inertial/mass').get('value')) for l in physical.values() if l.find('inertial/mass') is not None)
    for part in mapping['parts']:
        part['mass_kg'] = float(physical[part['link']].find('inertial/mass').get('value'))
    for part in mapping.get('contact_parts', []):
        part['mass_kg'] = float(physical[part['link']].find('inertial/mass').get('value'))
        if part.get('role') == 'pad_sleeve': part['width_scale'] = metadata['pad_width_scale']
    if code[0]: mapping['rigid_palm']['dimensions_m'][2] = metadata['palm_width_m']
    right.with_suffix('.mapping.json').write_text(json.dumps(mapping, indent=2))
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--baseline', required=True); p.add_argument('--code', required=True)
    p.add_argument('--output', required=True); p.add_argument('--compiler-dir', default='/home/zjh/Evolution_PC/Isaaclab_other')
    a = p.parse_args(); report = build(a.baseline, validate(json.loads(a.code)), a.output, a.compiler_dir)
    print(json.dumps(report, indent=2)); sys.exit(0 if report['passed'] else 2)
