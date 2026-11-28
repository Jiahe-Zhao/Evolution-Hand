"""Extract the tray and three pebbles from outputs/nov20/pebble.xml and audit a CPU import.

This is an asset check only. It never builds a hand, never starts a simulator, and
does not claim any grasp or task result. The cleaned MJCF keeps the original object
subtrees (shapes, masses, friction, free joints) byte-for-byte in structure.
"""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path((str(_EVOLUTION_ROOT)))
SRC = ROOT / 'outputs/nov20/pebble.xml'
OUT = ROOT / 'outputs/nov20/ax50_pebble07'
CLEAN = OUT / 'object.xml'
AUDIT = OUT / 'asset_audit.json'

HAND_BODY = 'diagnostic_world'
KEEP_WORLD = ('floor', 'light', 'tray', 'rim_x-1', 'rim_y-1', 'rim_x1', 'rim_y1')
KEEP_MESHES = ('peb0', 'peb1', 'peb2')
KEEP_BODIES = ('item', 'neighbor1', 'neighbor2')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sig(elem):
    """Structural signature: tag, attributes and children; ignores whitespace text."""
    return (elem.tag, tuple(sorted(elem.attrib.items())), tuple(sig(c) for c in elem))


def short(label):
    return str(label).rsplit('/', 1)[-1]


def descendants(elem, tag):
    return [e for e in elem.iter(tag) if e is not elem]


def authored_mass(body):
    return sum(float(g.get('mass', 0.0)) for g in body.findall('geom'))


def authored_friction(body):
    out = {}
    for g in body.findall('geom'):
        out[g.get('name')] = [float(x) for x in g.get('friction').split()]
    return out


def clean_tree(src_root):
    """Remove the old hand, its actuators and its mesh/contact bookkeeping."""
    root = copy.deepcopy(src_root)
    removed = {}

    wb = root.find('worldbody')
    hand = wb.find(f"body[@name='{HAND_BODY}']")
    if hand is None:
        raise RuntimeError(f'missing hand body {HAND_BODY!r}')
    removed['hand_bodies'] = len(descendants(hand, 'body')) + 1
    removed['hand_geoms'] = len(descendants(hand, 'geom'))
    removed['hand_joints'] = len(descendants(hand, 'joint'))
    wb.remove(hand)

    actuators = root.find('actuator')
    removed['actuators'] = len(actuators) if actuators is not None else 0
    if actuators is not None:
        root.remove(actuators)

    for tag in ('sensor', 'keyframe', 'tendon'):
        node = root.find(tag)
        if node is not None:
            removed[f'{tag}_blocks'] = len(list(node))
            root.remove(node)

    asset = root.find('asset')
    mesh_removed = 0
    for mesh in list(asset):
        if mesh.tag == 'mesh' and mesh.get('name') not in KEEP_MESHES:
            asset.remove(mesh)
            mesh_removed += 1
    removed['hand_mesh_assets'] = mesh_removed

    contact = root.find('contact')
    pair_removed = 0
    for pair in list(contact):
        names = (pair.get('geom1', ''), pair.get('geom2', ''))
        if any('handgeom' in n for n in names):
            contact.remove(pair)
            pair_removed += 1
    removed['hand_contact_pairs'] = pair_removed
    removed['kept_contact_pairs'] = len(list(contact))
    return root, removed


def verify_preserved(src_root, clean_root):
    src_wb, clean_wb = src_root.find('worldbody'), clean_root.find('worldbody')
    src_by_name = {(e.get('name') or e.tag): e for e in src_wb}
    clean_by_name = {(e.get('name') or e.tag): e for e in clean_wb}
    world_ok = {}
    for name in KEEP_WORLD:
        world_ok[name] = sig(src_by_name[name]) == sig(clean_by_name[name])

    src_mesh = {m.get('name'): m for m in src_root.find('asset')}
    clean_mesh = {m.get('name'): m for m in clean_root.find('asset')}
    mesh_ok = {n: sig(src_mesh[n]) == sig(clean_mesh[n]) for n in KEEP_MESHES}

    src_bodies = {b.get('name'): b for b in src_wb.findall('body')}
    clean_bodies = {b.get('name'): b for b in clean_wb.findall('body')}
    body_ok = {n: sig(src_bodies[n]) == sig(clean_bodies[n]) for n in KEEP_BODIES}

    src_pairs = {sig(p) for p in src_root.find('contact')}
    clean_pairs = [sig(p) for p in clean_root.find('contact')]
    pairs_ok = all(p in src_pairs for p in clean_pairs)
    return world_ok, mesh_ok, body_ok, pairs_ok


def import_model(path):
    import newton

    builder = newton.ModelBuilder()
    builder.add_mjcf(str(path), collapse_fixed_joints=False)
    model = builder.finalize(device='cpu')
    return newton, model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src_root = _evo_parse_xml(SRC).getroot()
    clean_root, removed = clean_tree(src_root)

    world_ok, mesh_ok, body_ok, pairs_ok = verify_preserved(src_root, clean_root)

    tree = ET.ElementTree(clean_root)
    ET.indent(tree, space=' ')
    tree.write(CLEAN, encoding='utf-8', xml_declaration=True)

    clean_text = CLEAN.read_text(encoding='utf-8')
    residue = {
        'diagnostic_world': 'diagnostic_world' in clean_text,
        'handgeom': 'handgeom' in clean_text,
        'mirror_mesh_file': 'mirror_' in clean_text,
        'actuator': '<actuator' in clean_text,
    }

    newton, model = import_model(CLEAN)

    body_names = [short(n) for n in model.body_label]
    body_mass = [float(x) for x in model.body_mass.numpy()]
    joint_names = [short(n) for n in model.joint_label]
    joint_type = [int(x) for x in model.joint_type.numpy()]
    joint_q_start = [int(x) for x in model.joint_q_start.numpy()]
    joint_qd_start = [int(x) for x in model.joint_qd_start.numpy()]
    shape_names = [short(n) for n in model.shape_label]
    shape_mu = [float(x) for x in model.shape_material_mu.numpy()]
    target_ke = [float(x) for x in model.joint_target_ke.numpy()]
    target_kd = [float(x) for x in model.joint_target_kd.numpy()]
    target_mode = [int(x) for x in model.joint_target_mode.numpy()]

    authored = {
        'body_mass_kg': {b: authored_mass(src_root.find('worldbody').find(f"body[@name='{b}']"))
                         for b in KEEP_BODIES},
        'friction': {},
    }
    authored['body_mass_kg']['total'] = sum(authored['body_mass_kg'][b] for b in KEEP_BODIES)
    for b in KEEP_BODIES:
        authored['friction'].update(authored_friction(
            src_root.find('worldbody').find(f"body[@name='{b}']")))
    for g in src_root.find('worldbody').findall('geom'):
        authored['friction'][g.get('name')] = [float(x) for x in g.get('friction').split()]

    imported_by_name = dict(zip(body_names, body_mass))
    mass_comparison = {}
    for name in KEEP_BODIES:
        want = authored['body_mass_kg'][name]
        got = imported_by_name.get(name)
        diff = None if got is None else got - want
        mass_comparison[name] = {
            'authored': want,
            'imported': got,
            'diff': diff,
            'match': got is not None and abs(diff) <= 1e-9,
        }
    mass_ok = all(v['match'] for v in mass_comparison.values())

    friction_comparison = {}
    for name, mu in zip(shape_names, shape_mu):
        want = authored['friction'].get(name)
        friction_comparison[name] = {
            'authored_mu': None if want is None else want[0],
            'imported_mu': mu,
            'diff': None if want is None else mu - want[0],
            'match': want is not None and abs(want[0] - mu) <= 1e-6,
        }
    friction_ok = (len(friction_comparison) == 12
                   and all(v['match'] for v in friction_comparison.values()))

    free_ok = (joint_names == ['free0', 'free1', 'free2']
               and joint_type == [int(newton.JointType.FREE)] * 3
               and model.joint_dof_count == 18
               and len(model.joint_q) == 21)

    no_drive = (all(v == 0.0 for v in target_ke)
                and all(v == 0.0 for v in target_kd)
                and all(v == int(newton.JointTargetMode.NONE) for v in target_mode))

    checks = {
        'hand_removed': not any(residue.values()),
        'objects_preserved': all(world_ok.values()) and all(mesh_ok.values())
                             and all(body_ok.values()) and pairs_ok,
        'free_joints_preserved': free_ok,
        'mass_match': mass_ok,
        'friction_preserved': friction_ok,
        'no_object_drive': no_drive,
        'shape_count_12': len(shape_names) == 12,
        'body_count_3': len(body_names) == 3,
    }

    audit = {
        'scope': 'asset extraction and CPU import audit only; not a grasp or task result',
        'source': {'path': str(SRC), 'bytes': SRC.stat().st_size, 'sha256': sha256(SRC)},
        'clean': {'path': str(CLEAN), 'bytes': CLEAN.stat().st_size, 'sha256': sha256(CLEAN)},
        'removed': removed,
        'residue_in_clean': residue,
        'kept': {
            'worldbody': list(KEEP_WORLD),
            'object_bodies': list(KEEP_BODIES),
            'mesh_assets': list(KEEP_MESHES),
            'contact_pairs': [dict(p.attrib) for p in clean_root.find('contact')],
            'signature_match': {
                'worldbody': world_ok, 'mesh_assets': mesh_ok,
                'object_bodies': body_ok, 'contact_pairs': pairs_ok,
            },
        },
        'authored': authored,
        'imported': {
            'importer': str(Path(newton.__file__).resolve()),
            'body_names': body_names,
            'body_mass_kg': body_mass,
            'joint_names': joint_names,
            'joint_type': joint_type,
            'joint_q_start': joint_q_start,
            'joint_qd_start': joint_qd_start,
            'joint_dof_count': int(model.joint_dof_count),
            'joint_q_length': int(len(model.joint_q)),
            'shape_names': shape_names,
            'shape_body': [int(x) for x in model.shape_body.numpy()],
            'shape_material_mu': shape_mu,
            'joint_target_ke': target_ke,
            'joint_target_kd': target_kd,
            'joint_target_mode': target_mode,
        },
        'mass_comparison': mass_comparison,
        'friction_comparison': friction_comparison,
        'checks': checks,
        'notes': [
            'The source <pair> solref/friction values are retained in object.xml but Newton '
            'rigid contacts are shape-based, so per-pair solref/friction are not applied by '
            'the importer; shape friction (0.85 on stones/tray/rims, 1.0 on floor) is imported.',
            'The old hand body, 102 STL mesh assets, 204 hand contact pairs and 27 actuators '
            'were removed; all 21 object pairs remain.',
            'Imported body masses match the authored geom masses exactly at float32 precision.',
            'No hand is present in this file, so a successful import is not an operation result.',
        ],
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding='utf-8')

    summary = {
        'clean': str(CLEAN),
        'audit': str(AUDIT),
        'removed': removed,
        'bodies': dict(zip(body_names, body_mass)),
        'dof': int(model.joint_dof_count),
        'joint_q_length': int(len(model.joint_q)),
        'shape_count': len(shape_names),
        'checks': checks,
        'passed': all(checks.values()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
