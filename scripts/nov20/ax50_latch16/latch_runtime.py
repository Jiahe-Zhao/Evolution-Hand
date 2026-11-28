"""AX50 v3 hand + cleaned latch-16 passive box/lid/bolt mechanism."""

from __future__ import annotations

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path((str(_EVOLUTION_ROOT)))
sys.path.insert(0, str(ROOT / 'outputs/ax50_visualskin_v3_0_0'))
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_keys17'))

import ax50_visualskin  # noqa: F401,E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

import damped_adapter  # noqa: E402
import pair_damping  # noqa: E402

SRC_XML = ROOT / 'outputs/nov20/latch16/latch16.xml'
OUT_DIR = ROOT / 'outputs/nov20/ax50_latch16'
CLEAN_XML = OUT_DIR / 'latch_clean.xml'


@wp.kernel
def wrist_pd(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    target: wp.array(dtype=float),
    caps: wp.array(dtype=float),
    idx: wp.array(dtype=int),
    force: wp.array(dtype=float),
):
    i = wp.tid()
    j = idx[i]
    kp = 4.0
    kd = 0.08
    cap = caps[j]
    if j < 3:
        kp = 1000.0
        kd = 25.0
        cap = 3.0
    elif j < 6:
        kp = 20.0
        kd = 0.5
        cap = 2.0
    force[j] = wp.clamp(kp * (target[j] - q[j]) - kd * qd[j], -cap, cap)


damped_adapter.control_pd_subset = wrist_pd


@wp.kernel
def set_prop_damping(
    count: wp.array(dtype=int),
    shape: wp.array(dtype=int),
    is_prop: wp.array(dtype=int),
    kd: float,
    values: wp.array(dtype=float),
):
    i = wp.tid()
    if i < count[0] and is_prop[shape[i]] != 0:
        values[i] = kd


class LatchVBD(pair_damping.KeyPairVBD):
    """KeyPairVBD plus latch-16 prop soft-contact damping override."""

    def _refresh_body_particle_contact_state(self, contacts, refresh):
        super()._refresh_body_particle_contact_state(contacts, refresh)
        for shape in getattr(self, 'extra_palm_shapes', []):
            wp.launch(
                pair_damping.set_pair_damping,
                contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    shape,
                    self.pair_palm_count,
                    0.5,
                    self.body_particle_contact_material_kd,
                ],
            )
        flags = getattr(self, 'prop_shape_flags', None)
        if contacts is not None and flags is not None:
            wp.launch(
                set_prop_damping,
                contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    flags,
                    self.prop_contact_kd,
                    self.body_particle_contact_material_kd,
                ],
            )
            override_ke = getattr(self, 'prop_contact_ke', None)
            if override_ke is not None:
                wp.launch(
                    set_prop_damping,
                    contacts.soft_contact_max,
                    inputs=[
                        contacts.soft_contact_count,
                        contacts.soft_contact_shape,
                        flags,
                        override_ke,
                        self.body_particle_contact_material_ke,
                    ],
                )


pair_damping.KeyPairVBD = LatchVBD


def clean_latch_xml(src=SRC_XML, dst=CLEAN_XML):
    """Strip legacy hand_111 subtree/actuators/contact pairs, keep box/lid/bolt."""
    src = Path(src)
    dst = Path(dst)
    tree = _evo_parse_xml(src)
    root = tree.getroot()

    removed = {}
    act = root.find('actuator')
    if act is not None:
        removed['actuator'] = len(list(act))
        root.remove(act)

    world = root.find('worldbody')
    diag = world.find("./body[@name='diagnostic_world']")
    if diag is not None:
        removed['diagnostic_world_bodies'] = len(list(diag.iter('body')))
        world.remove(diag)

    asset = root.find('asset')
    dropped_assets = 0
    if asset is not None:
        for child in list(asset):
            if child.tag == 'mesh':
                dropped_assets += 1
                asset.remove(child)
    removed['legacy_mesh_assets'] = dropped_assets

    contact = root.find('contact')
    dropped_pairs = 0
    kept_pairs = []
    if contact is not None:
        for pair in list(contact):
            if pair.tag != 'pair':
                continue
            g1 = pair.attrib.get('geom1', '')
            g2 = pair.attrib.get('geom2', '')
            if g1.startswith('handgeom') or g2.startswith('handgeom'):
                contact.remove(pair)
                dropped_pairs += 1
            else:
                kept_pairs.append((g1, g2))
    removed['hand_contact_pairs'] = dropped_pairs

    ET.indent(tree, space=' ')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding='utf-8', xml_declaration=True)

    return {
        'source': str(src),
        'clean': str(dst),
        'removed': removed,
        'kept_contact_pairs': kept_pairs,
        'kept_world_geoms': [
            c.attrib.get('name') for c in world if c.tag in ('geom', 'camera', 'light')
        ],
        'kept_bodies': [b.attrib.get('name') for b in world.iter('body')],
    }


class LatchRuntime(damped_adapter.AX50BrushRuntime):
    """AX50 hand + passive box/lid/bolt; lid_hinge and slide_bolt are purely passive."""

    def __init__(self, *args, latch_xml=None, **kwargs):
        self.source_xml = Path(latch_xml or SRC_XML)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.clean_xml = CLEAN_XML
        self.clean_summary = clean_latch_xml(self.source_xml, self.clean_xml)
        super().__init__(*args, **kwargs)

    def _add_brush(self, b):
        b.default_joint_cfg.target_ke = 0.0
        b.default_joint_cfg.target_kd = 0.0
        from newton._src.sim.builder import _BuilderShapeCollisionFilterPairs
        if isinstance(b._shape_collision_filter_pairs, list):
            previous = list(b._shape_collision_filter_pairs)
            restored = _BuilderShapeCollisionFilterPairs()
            for pair in previous:
                restored.append(pair)
            b._shape_collision_filter_pairs = restored
        b.add_mjcf(str(self.clean_xml), collapse_fixed_joints=False)

        self.prop_body_count = int(b.body_count - self.hand_body_count)
        labels = [str(x) for x in b.body_label]
        joints = [str(x) for x in b.joint_label]

        self.lid_body = next(i for i, n in enumerate(labels) if n.rsplit('/', 1)[-1] == 'lid')
        self.bolt_body = next(i for i, n in enumerate(labels) if n.rsplit('/', 1)[-1] == 'bolt')
        self.prop_bodies = [self.lid_body, self.bolt_body]

        self.latch_joints = {}
        for j, name in enumerate(joints):
            short = name.rsplit('/', 1)[-1]
            if short in ('slide_bolt', 'lid_hinge'):
                q = int(b.joint_q_start[j])
                dof = int(b.joint_qd_start[j])
                b.joint_target_ke[dof] = 0.0
                b.joint_target_kd[dof] = 0.0
                b.joint_target_mode[dof] = newton.JointTargetMode.NONE
                self.latch_joints[short] = {'joint': j, 'q_start': q, 'dof': dof}

        self.slide_q = self.latch_joints['slide_bolt']['q_start']
        self.lid_q = self.latch_joints['lid_hinge']['q_start']
        self.slide_dof = self.latch_joints['slide_bolt']['dof']
        self.lid_dof = self.latch_joints['lid_hinge']['dof']

    def _build(self, iterations, indent, gravity, scene_builder):
        super()._build(iterations, indent, gravity, scene_builder)
        m = self.model
        shape_body = m.shape_body.numpy()
        # Mark all shapes belonging to lid, bolt, or world static box geoms (body == -1)
        flags = np.array(
            [1 if (int(b) in (self.lid_body, self.bolt_body) or int(b) < 0) else 0 for b in shape_body],
            dtype=int,
        )
        self.prop_shape_flags_np = flags
        self.vbd.extra_palm_shapes = [
            i for i, label in enumerate(m.shape_label)
            if str(label).endswith(('/link_2_0/palm_tile_12', '/link_2_0/palm_tile_14'))
        ]
        self.vbd.prop_shape_flags = wp.array(flags, dtype=int)
        self.vbd.prop_contact_kd = float(os.environ.get('LATCH_KD', '0.5'))

    def latch_info(self):
        m = self.model
        mass = m.body_mass.numpy()
        damping = m.joint_damping.numpy()
        armature = m.joint_armature.numpy()
        lo = m.joint_limit_lower.numpy()
        hi = m.joint_limit_upper.numpy()
        ke = m.joint_target_ke.numpy()
        kd = m.joint_target_kd.numpy()
        mode = m.joint_target_mode.numpy()
        joints = {}
        for name, row in self.latch_joints.items():
            d = row['dof']
            joints[name] = {
                'q_start': row['q_start'],
                'dof': d,
                'damping': float(damping[d]),
                'armature': float(armature[d]),
                'limit_lower': float(lo[d]),
                'limit_upper': float(hi[d]),
                'target_ke': float(ke[d]),
                'target_kd': float(kd[d]),
                'target_mode': int(mode[d]),
            }
        return {
            'source_xml': str(self.source_xml),
            'clean_xml': str(self.clean_xml),
            'clean_summary': self.clean_summary,
            'hand_body_count': int(self.hand_body_count),
            'prop_body_count': int(self.prop_body_count),
            'total_body_count': int(m.body_count),
            'lid_body': int(self.lid_body),
            'bolt_body': int(self.bolt_body),
            'lid_mass_kg': float(mass[self.lid_body]),
            'bolt_mass_kg': float(mass[self.bolt_body]),
            'latch_joints': joints,
            'prop_contact_kd': float(getattr(self.vbd, 'prop_contact_kd', float('nan'))),
        }
