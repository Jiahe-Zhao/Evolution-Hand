"""AX50 v3 hand plus the cleaned leaf-04 passive object (model derivation only).

Scope: add the object subtree of ``outputs/nov20/leaf04/hand.xml`` to the
existing AX50 shared-world builder. The legacy hand_111 subtree, its actuators,
its mesh assets and its hand/leaf contact exclusions are stripped before the
MJCF is imported. Leaf DOFs are never position-driven: the hand PD kernel only
writes hand DOF indices, and a dedicated kernel writes the passive leaf torque
``-0.0014*q - 0.04*(q - clamp(q, 0, 1.58))`` into ``control.joint_f`` each step.
The MJCF joint damping (0.00032) and limits (0..1.58 rad) stay native.

The leaf soft-contact damping (kd=.5) is injected through ``LeafVBD`` and its
own ``leaf_shape_flags`` attribute; ``pair_damping.KeyPairVBD`` and the keys17
files are untouched.
"""

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
sys.path.insert(0, str(ROOT / "outputs/ax50_visualskin_v3_0_0"))
sys.path.insert(0, str(ROOT / "scripts/nov20/ax50_keys17"))

import ax50_visualskin  # noqa: F401,E402  -- must be imported before newton
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

import damped_adapter  # noqa: E402
import pair_damping  # noqa: E402

SRC_XML = ROOT / "outputs/nov20/leaf04/hand.xml"
OUT_DIR = ROOT / "outputs/nov20/ax50_leaf04"
CLEAN_XML = OUT_DIR / "leaf_clean.xml"

LEAF_DAMPING = 0.00032
LEAF_ARMATURE = 1.0e-6
LEAF_LIMIT_LOW = 0.0
LEAF_LIMIT_HIGH = 1.58


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
        kp = 3000.0
        kd = 35.0
        cap = 30.0
    elif j < 6:
        kp = 20.0
        kd = 0.5
        cap = 2.0
    force[j] = wp.clamp(kp * (target[j] - q[j]) - kd * qd[j], -cap, cap)


# The hand PD in damped_adapter.step resolves this module global at call time.
damped_adapter.control_pd_subset = wrist_pd


@wp.kernel
def leaf_passive(
    q: wp.array(dtype=float),
    force: wp.array(dtype=float),
    idx: wp.array(dtype=int),
):
    i = wp.tid()
    j = idx[i]
    force[j] = -0.0014 * q[j] - 0.04 * (q[j] - wp.clamp(q[j], 0.0, 1.58))


@wp.kernel
def add_leaf_force(
    force: wp.array(dtype=float),
    idx: wp.array(dtype=int),
    extra: wp.array(dtype=float),
):
    i = wp.tid()
    force[idx[i]] = force[idx[i]] + extra[i]


@wp.kernel
def set_leaf_damping(
    count: wp.array(dtype=int),
    shape: wp.array(dtype=int),
    is_leaf: wp.array(dtype=int),
    kd: float,
    values: wp.array(dtype=float),
):
    i = wp.tid()
    if i < count[0] and is_leaf[shape[i]] != 0:
        values[i] = kd


class LeafVBD(pair_damping.KeyPairVBD):
    """KeyPairVBD plus a leaf-only soft-contact damping override."""

    def _refresh_body_particle_contact_state(self, contacts, refresh):
        super()._refresh_body_particle_contact_state(contacts, refresh)
        for shape in getattr(self, 'extra_palm_shapes', []):
            wp.launch(pair_damping.set_pair_damping, contacts.soft_contact_max, inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric, contacts.soft_contact_shape, shape, self.pair_palm_count, 0.5, self.body_particle_contact_material_kd])
        flags = getattr(self, "leaf_shape_flags", None)
        if contacts is not None and flags is not None:
            wp.launch(
                set_leaf_damping,
                contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    flags,
                    self.leaf_contact_kd,
                    self.body_particle_contact_material_kd,
                ],
            )
            override_ke = getattr(self, "leaf_contact_ke", None)
            if override_ke is not None:
                wp.launch(set_leaf_damping, contacts.soft_contact_max,
                    inputs=[contacts.soft_contact_count, contacts.soft_contact_shape,
                            flags, override_ke, self.body_particle_contact_material_ke])


# damped_adapter imports KeyPairVBD lazily inside _build, so patch the module.
pair_damping.KeyPairVBD = LeafVBD


def clean_leaf_xml(src=SRC_XML, dst=CLEAN_XML):
    """Strip the legacy hand subtree/actuators/contact pairs, keep the leaf object."""
    src = Path(src)
    dst = Path(dst)
    tree = _evo_parse_xml(src)
    root = tree.getroot()

    removed = {}
    act = root.find("actuator")
    if act is not None:
        removed["actuator"] = len(list(act))
        root.remove(act)

    world = root.find("worldbody")
    diag = world.find("./body[@name='diagnostic_world']")
    if diag is not None:
        removed["diagnostic_world_bodies"] = len(list(diag.iter("body")))
        world.remove(diag)

    asset = root.find("asset")
    keep_mesh = {"bladeMesh0", "bladeMesh1"}
    dropped_assets = 0
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh" and child.attrib.get("name") in keep_mesh:
                continue
            if child.tag == "mesh":
                dropped_assets += 1
            asset.remove(child)
    removed["legacy_mesh_assets"] = dropped_assets

    contact = root.find("contact")
    dropped_pairs = 0
    kept_pairs = []
    if contact is not None:
        for pair in list(contact):
            if pair.tag != "pair":
                continue
            g1 = pair.attrib.get("geom1", "")
            g2 = pair.attrib.get("geom2", "")
            if g1.startswith("handgeom") or g2.startswith("handgeom"):
                contact.remove(pair)
                dropped_pairs += 1
            else:
                kept_pairs.append((g1, g2))
    removed["hand_contact_pairs"] = dropped_pairs

    ET.indent(tree, space=" ")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)

    summary = {
        "source": str(src),
        "clean": str(dst),
        "removed": removed,
        "kept_contact_pairs": kept_pairs,
        "kept_world_geoms": [
            c.attrib.get("name")
            for c in world
            if c.tag in ("geom", "camera", "light")
        ],
        "kept_leaf_bodies": [b.attrib.get("name") for b in world.iter("body")],
    }
    return summary


class LeafRuntime(damped_adapter.AX50BrushRuntime):
    """AX50 hand + two hinged leaves; no position drive on the leaves."""

    def __init__(self, *args, leaf_xml=None, **kwargs):
        self.source_xml = Path(leaf_xml or SRC_XML)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.clean_xml = CLEAN_XML
        self.clean_summary = clean_leaf_xml(self.source_xml, self.clean_xml)
        self.extra_leaf_force = None
        super().__init__(*args, **kwargs)

    def _add_brush(self, b):
        b.default_joint_cfg.target_ke = 0.0
        b.default_joint_cfg.target_kd = 0.0
        # USD import materializes this pinned Newton version's filter manager as a list.
        # Restore its original representation without removing a single filter pair.
        from newton._src.sim.builder import _BuilderShapeCollisionFilterPairs
        if isinstance(b._shape_collision_filter_pairs,list):
            previous=list(b._shape_collision_filter_pairs)
            restored=_BuilderShapeCollisionFilterPairs()
            for pair in previous:restored.append(pair)
            b._shape_collision_filter_pairs=restored
        b.add_mjcf(str(self.clean_xml), collapse_fixed_joints=False)

        self.prop_body_count = int(b.body_count - self.hand_body_count)
        labels = [str(x) for x in b.body_label]
        joints = [str(x) for x in b.joint_label]

        self.leaf_bodies = [
            i for i, name in enumerate(labels) if name.rsplit("/", 1)[-1] in ("leaf0", "leaf1")
        ]
        self.leaf_body_count = len(self.leaf_bodies)

        self.leaf_joint_names = []
        self.leaf_q_indices = []
        self.leaf_dof_indices = []
        self.leaf_joint_rows = []
        for j, name in enumerate(joints):
            short = name.rsplit("/", 1)[-1]
            if short not in ("hinge0", "hinge1"):
                continue
            dof = int(b.joint_qd_start[j])
            q = int(b.joint_q_start[j])
            b.joint_target_mode[dof] = newton.JointTargetMode.NONE
            b.joint_target_ke[dof] = 0.0
            b.joint_target_kd[dof] = 0.0
            b.joint_damping[dof] = LEAF_DAMPING
            b.joint_armature[dof] = LEAF_ARMATURE
            b.joint_limit_lower[dof] = LEAF_LIMIT_LOW
            b.joint_limit_upper[dof] = LEAF_LIMIT_HIGH
            self.leaf_joint_names.append(short)
            self.leaf_q_indices.append(q)
            self.leaf_dof_indices.append(dof)
            self.leaf_joint_rows.append(
                {"name": short, "joint": j, "q_start": q, "dof": dof}
            )
        self.leaf_dof_count = len(self.leaf_dof_indices)
        self.leaf_dof_idx = wp.array(np.array(self.leaf_dof_indices, dtype=int), dtype=int)
        self.leaf_q_np = np.array(self.leaf_q_indices, dtype=int)
        self.leaf_dof_np = np.array(self.leaf_dof_indices, dtype=int)

        if self.leaf_body_count != 2 or self.leaf_dof_count != 2:
            raise RuntimeError(
                "leaf MJCF import mismatch: bodies=%r joints=%r"
                % (self.leaf_bodies, self.leaf_joint_rows)
            )
        print("[leaf] prop bodies added:", self.prop_body_count)
        print("[leaf] bodies:", [labels[i] for i in self.leaf_bodies])
        print("[leaf] joints:", self.leaf_joint_rows)

    def _build(self, iterations, indent, gravity, scene_builder):
        super()._build(iterations, indent, gravity, scene_builder)
        m = self.model
        shape_body = m.shape_body.numpy()
        leaf_set = set(int(i) for i in self.leaf_bodies)
        flags = np.array(
            [1 if int(b) in leaf_set else 0 for b in shape_body], dtype=int
        )
        self.leaf_shape_flags_np = flags
        self.leaf_shape_indices = np.flatnonzero(flags)
        self.leaf_shape_count = int(flags.sum())
        self.vbd.extra_palm_shapes = [i for i,label in enumerate(m.shape_label) if str(label).endswith(('/link_2_0/palm_tile_12','/link_2_0/palm_tile_14'))]
        self.vbd.leaf_shape_flags = wp.array(flags, dtype=int)
        self.vbd.leaf_contact_kd = float(os.environ.get("LEAF_KD", "0.5"))
        print(
            "[leaf] leaf shapes:",
            int(self.leaf_shape_count),
            "of",
            int(m.shape_count),
            "shape_count",
        )

    def step(self, dt=0.00025):
        wp.launch(
            leaf_passive,
            self.leaf_dof_count,
            inputs=[self.state.joint_q, self.control.joint_f, self.leaf_dof_idx],
        )
        if self.extra_leaf_force is not None:
            wp.launch(
                add_leaf_force,
                self.leaf_dof_count,
                inputs=[self.control.joint_f, self.leaf_dof_idx, self.extra_leaf_force],
            )
        super().step(dt)

    def leaf_info(self):
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
        for row in self.leaf_joint_rows:
            d = row["dof"]
            joints[row["name"]] = {
                "q_start": row["q_start"],
                "dof": d,
                "damping": float(damping[d]),
                "armature": float(armature[d]),
                "limit_lower": float(lo[d]),
                "limit_upper": float(hi[d]),
                "target_ke": float(ke[d]),
                "target_kd": float(kd[d]),
                "target_mode": int(mode[d]),
            }
        leaf_body_mass = [float(mass[i]) for i in self.leaf_bodies]
        vbd_counts = getattr(self.vbd, "body_particle_contact_counts", None)
        return {
            "source_xml": str(self.source_xml),
            "clean_xml": str(self.clean_xml),
            "clean_summary": self.clean_summary,
            "hand_body_count": int(self.hand_body_count),
            "prop_body_count": int(self.prop_body_count),
            "total_body_count": int(m.body_count),
            "leaf_body_count": int(self.leaf_body_count),
            "leaf_body_indices": [int(i) for i in self.leaf_bodies],
            "leaf_body_names": [
                str(m.body_label[i]).rsplit("/", 1)[-1] for i in self.leaf_bodies
            ],
            "leaf_body_mass_kg": leaf_body_mass,
            "leaf_body_mass_total_kg": float(sum(leaf_body_mass)),
            "hand_mass_kg": float(mass[: self.hand_body_count].sum()),
            "total_mass_kg": float(mass.sum()),
            "leaf_joints": joints,
            "shape_count": int(m.shape_count),
            "leaf_shape_count": int(self.leaf_shape_count),
            "leaf_shape_indices": [int(i) for i in self.leaf_shape_indices],
            "pair_shape": int(getattr(self.vbd, "pair_shape", -1)),
            "pair_kd": float(getattr(self.vbd, "pair_kd", float("nan"))),
            "leaf_contact_kd": float(getattr(self.vbd, "leaf_contact_kd", float("nan"))),
            "leaf_contact_ke_override": getattr(self.vbd, "leaf_contact_ke", None),
            "vbd_rigid_body_particle_contact_buffer_size": 32768,
            "vbd_body_particle_contact_counts_len": (
                int(vbd_counts.shape[0]) if vbd_counts is not None else None
            ),
            "soft_contact_max": int(self.contacts.soft_contact_max),
            "rigid_contact_max": int(self.rigid_observation.rigid_contact_max),
            "dof_count": int(m.joint_dof_count),
            "joint_q_length": int(m.joint_q.shape[0]),
        }
