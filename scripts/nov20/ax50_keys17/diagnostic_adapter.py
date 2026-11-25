"""Derived AX50 VisualSkin v3.0.0 runtime with a free 13x5 bristle brush.

Scope: shared-world adapter + smoke only. The hand still uses the packaged
MuJoCo-Warp rigid solver and the packaged VBD soft palm/pads. The brush is an
extra articulation in the same builder, extracted from
``outputs/nov20/brush14/verified.xml`` (the brush subtree only; the legacy
hand111 subtree is never imported).

Brush bending is a native MuJoCo passive spring
(``mujoco:dof_passive_stiffness``, MJCF ``stiffness``) plus native joint
damping, exactly as ``scripts/nov20/brush14/run_brush.py`` configures it.
Brush DOFs are deliberately excluded from the hand PD kernel.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import ax50_visualskin  # noqa: F401  -- must be imported before newton
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from pxr import Usd, UsdPhysics as P

from ax50_visualskin import HandRuntime
from ax50_visualskin.runtime import (
    add_feedback,
    anchors,
    contact_reaction,
    filter_bridge_contacts,
    sync_ghosts,
)
from ax50_visualskin.sensors import collect_rigid_load, collect_soft_load, record_peak

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BRUSH_XML = REPO_ROOT / "outputs/nov20/brush14/verified.xml"

# run_brush.py: mm.jnt_stiffness[ji] = EI / (0.022 / 5); dof_damping = 2e-7.
BRUSH_HINGE_STIFFNESS = 4.545e-4  # N m / rad
BRUSH_HINGE_DAMPING = 2e-7  # N m s / rad
BRUSH_SPRINGREF = 0.0

# run_brush.py pair overrides. Newton rigid contacts are per-shape, not per-pair,
# so each shape carries the coefficient of its dominant pair.
SHAFT_MU = 1.1  # brush/hand condim 6
SHAFT_MU_TORSIONAL = 0.004
SHAFT_MU_ROLLING = 1.0e-4
BRISTLE_MU = 0.45  # bristle/paper condim 3
BRISTLE_MU_TORSIONAL = 1.0e-5
BRISTLE_MU_ROLLING = 1.0e-4
PAPER_MU = 0.5  # brush/paper condim 3


@wp.kernel
def control_pd_subset(
    q: wp.array(dtype=float),
    qd: wp.array(dtype=float),
    target: wp.array(dtype=float),
    caps: wp.array(dtype=float),
    idx: wp.array(dtype=int),
    force: wp.array(dtype=float),
):
    """Same law as the packaged control_pd, applied only to the hand DOFs."""
    i = wp.tid()
    j = idx[i]
    force[j] = wp.clamp(4.0 * (target[j] - q[j]) - 0.08 * qd[j], -caps[j], caps[j])


def _floats(text, count, default=0.0):
    if text is None:
        return (default,) * count
    values = [float(v) for v in text.split()]
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {text!r}")
    return tuple(values)


def _quat_from_body(node):
    """Body-local rotation. MuJoCo euler is degrees, default sequence xyz."""
    if node.get("quat") is not None:
        w, x, y, z = _floats(node.get("quat"), 4)
        return wp.normalize(wp.quat(x, y, z, w))
    if node.get("euler") is not None:
        ex, ey, ez = (math.radians(v) for v in _floats(node.get("euler"), 3))
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), ex)
        qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), ey)
        qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ez)
        return wp.normalize(wp.mul(wp.mul(qx, qy), qz))
    if node.get("axisangle") is not None:
        ax, ay, az, angle = _floats(node.get("axisangle"), 4)
        return wp.quat_from_axis_angle(wp.normalize(wp.vec3(ax, ay, az)), math.radians(angle))
    return wp.quat_identity()


def _local_xform(node):
    return wp.transform(wp.vec3(*_floats(node.get("pos"), 3)), _quat_from_body(node))


def _children(node, tag):
    return [c for c in node if c.tag == tag]


def _one_child(node, tag, name=None):
    for c in node:
        if c.tag == tag and (name is None or c.get("name") == name):
            return c
    return None


def parse_brush_xml(path):
    """Extract the brush subtree and the paper/table geoms; ignore hand111."""
    root = ET.parse(str(path)).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{path}: no <worldbody>")
    brush = _one_child(world, "body", "brush")
    if brush is None:
        raise ValueError(f"{path}: no <body name='brush'>")
    paper = _one_child(world, "geom", "paper")
    table = _one_child(world, "geom", "table")

    root_x = _local_xform(brush)
    shaft = _one_child(brush, "geom", "shaft")
    if shaft is None:
        raise ValueError(f"{path}: brush has no shaft geom")
    f0 = np.array(_floats(shaft.get("fromto"), 6)[:3])
    f1 = np.array(_floats(shaft.get("fromto"), 6)[3:])
    shaft_center = (f0 + f1) / 2.0
    shaft_half = float(np.linalg.norm(f1 - f0)) / 2.0
    shaft_radius = _floats(shaft.get("size"), 1)[0]
    shaft_mass = float(shaft.get("mass"))

    tuft = _one_child(brush, "body", "tuft_mount")
    if tuft is None:
        raise ValueError(f"{path}: brush has no tuft_mount")
    tuft_x = wp.transform_multiply(root_x, _local_xform(tuft))

    bundles = []
    for bundle in _children(tuft, "body"):
        segments = []
        node = bundle
        parent_x = tuft_x
        while node is not None:
            x = wp.transform_multiply(parent_x, _local_xform(node))
            inertial = _one_child(node, "inertial")
            capsule = None
            for g in _children(node, "geom"):
                if g.get("type") == "capsule":
                    capsule = g
            if inertial is None or capsule is None:
                raise ValueError(f"{path}: malformed bristle segment {node.get('name')}")
            lo, hi = _floats(capsule.get("fromto"), 6)[:3], _floats(capsule.get("fromto"), 6)[3:]
            seg_half = float(np.linalg.norm(np.subtract(hi, lo))) / 2.0
            segments.append(
                {
                    "name": node.get("name"),
                    "xform": x,
                    "com": wp.vec3(*_floats(inertial.get("pos"), 3)),
                    "mass": float(inertial.get("mass")),
                    "inertia": np.array(_floats(inertial.get("diaginertia"), 3)),
                    "radius": _floats(capsule.get("size"), 1)[0],
                    "half_height": seg_half,
                }
            )
            node = _one_child(node, "body")
            parent_x = x
        bundles.append(segments)

    def prop(node, name, collide, mu):
        if node is None:
            return None
        return {
            "name": name,
            "xform": wp.transform(wp.vec3(*_floats(node.get("pos"), 3)), wp.quat_identity()),
            "half": _floats(node.get("size"), 3),
            "collide": collide,
            "mu": mu,
        }

    return {
        "path": str(path),
        "root_xform": root_x,
        "shaft": {
            "center": wp.vec3(*shaft_center),
            "half_height": shaft_half,
            "radius": shaft_radius,
            "mass": shaft_mass,
        },
        "bundles": bundles,
        "paper": prop(paper, "paper", True, PAPER_MU),
        "table": prop(table, "table", False, 0.0),
    }


def _cylinder_inertia(mass, radius, length):
    ixx = mass * (3.0 * radius * radius + length * length) / 12.0
    izz = mass * radius * radius / 2.0
    return np.array([ixx, ixx, izz])


class AX50BrushRuntime(HandRuntime):
    """HandRuntime with one extra free brush articulation in the same world."""

    def __init__(self, *args, brush_xml=None, **kwargs):
        self.brush_xml = Path(brush_xml or os.environ.get("AX50_BRUSH_XML") or DEFAULT_BRUSH_XML)
        super().__init__(*args, **kwargs)

    def _build(self, iterations, indent, gravity, scene_builder):
        if indent:
            raise NotImplementedError("fixed_skeleton=True is not supported with the brush adapter")
        if scene_builder is not None:
            raise NotImplementedError(
                "scene_builder is not supported here; add props inside _add_brush instead"
            )
        variant = "pads"
        self.bridges = dict(np.load(self.asset / "bridges.npz"))
        self.pads = dict(np.load(self.asset / "pads.npz"))
        self.palm = dict(np.load(self.asset / "membrane.npz"))
        spec = json.loads((self.asset / "slidingskin.json").read_text())
        self.convention = json.loads((self.asset / "joint_convention.json").read_text())[
            "canonical_to_native"
        ]
        st = Usd.Stage.Open(str(self.asset / "hand.usda"))
        st.SetEditTarget(st.GetSessionLayer())
        if self.wrist == "locked":
            for p in list(st.Traverse()):
                if p.IsA(P.Joint) and p.GetName().startswith(("wrist_", "diagnostic_wrist_")):
                    P.FixedJoint.Define(st, p.GetPath())
        mount = wp.transform(self.position, self.orientation)
        b = newton.ModelBuilder(gravity=gravity)
        b.add_usd(st, xform=mount, collapse_fixed_joints=False, enable_self_collisions=True)
        b.rigid_gap = 0.0
        b.shape_gap[:] = [0.0] * len(b.shape_gap)
        self.body_names = [x.split("/")[-1] for x in b.body_label]
        self.hand_body_count = b.body_count
        self.joint_names = [""] * len(b.joint_target_ke)
        for j, label in enumerate(b.joint_label):
            if b.joint_type[j] == newton.JointType.FIXED:
                continue
            i = b.joint_qd_start[j]
            self.joint_names[i] = label.split("/")[-1]
            b.joint_target_mode[i] = newton.JointTargetMode.NONE
            b.joint_target_ke[i] = 0.0
            b.joint_target_kd[i] = 0.0
            b.joint_armature[i] = 0.00008
        self.hand_dof_count = len(b.joint_target_ke)
        # Declared after the USD import so mjc parsing of the packaged assets is untouched.
        SolverMuJoCo.register_custom_attributes(b)
        self._add_brush(b)
        self.probe = -999
        # Keep the approved palm geometry and its 3% runtime bending damping.
        z = self.palm
        self.palm_count = len(z["surface_rest_vertices_m"])
        b.add_cloth_mesh(
            pos=self.position,
            rot=wp.quat(*self.orientation),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=z["surface_rest_vertices_m"],
            indices=z["surface_faces"].flatten(),
            density=spec["surface_density_kg_m2"],
            tri_ke=spec["tri_ke"],
            tri_ka=spec["tri_ka"],
            tri_kd=spec["tri_kd"],
            edge_ke=spec["edge_ke"],
            edge_kd=spec["edge_kd"] * 0.03,
            particle_radius=spec["particle_radius_m"],
        )
        self.pad_offset = b.particle_count
        z = self.pads
        b.add_soft_mesh(
            pos=self.position,
            rot=wp.quat(*self.orientation),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=z["vertices"],
            indices=z["tets"].flatten(),
            density=1000.0,
            k_mu=5000.0,
            k_lambda=10000.0,
            k_damp=0.02,
            particle_radius=0.00005,
            add_surface_mesh_edges=True,
        )
        self.bridge_offset = b.particle_count
        self.bridge_tri_offset = b.tri_count
        self.bridge_edge_offset = b.edge_count
        z = self.bridges
        b.add_cloth_mesh(
            pos=self.position,
            rot=wp.quat(*self.orientation),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=z["vertices"],
            indices=z["faces"].flatten(),
            density=0.35,
            tri_ke=2.0,
            tri_ka=2.0,
            tri_kd=0.0001,
            edge_ke=2e-8,
            edge_kd=6e-10,
            particle_radius=0.00005,
        )
        self.real_count = b.particle_count
        self.bridge_tri_end = b.tri_count
        self.bridge_edge_end = b.edge_count
        interior = (
            np.setdiff1d(np.arange(len(self.pads["vertices"])), np.unique(self.pads["faces"]))
            + self.pad_offset
        ).tolist()
        z = self.palm
        pis = list(z["anchor_particle"])
        bis = [self.body_names.index(str(z["bone_names"][j])) for j in z["anchor_bone"]]
        loc = list(z["anchor_local_m"])
        ks = list(z["anchor_stiffness_N_m"])
        z = self.pads
        pis += list(z["anchor_particle"] + self.pad_offset)
        bis += [self.body_names.index(str(z["bone_names"][j])) for j in z["anchor_bone"]]
        loc += list(z["anchor_local"])
        ks += [100.0] * len(z["anchor_particle"])
        z = self.bridges
        pis += list(z["root_pi"] + self.bridge_offset)
        bis += [self.body_names.index(str(self.pads["bone_names"][i])) for i in z["root_bone"]]
        loc += list(z["root_local"])
        ks += [10.0] * len(z["root_pi"])
        ghost = []
        for pi, bi, ll, k in zip(pis, bis, loc, ks):
            x = wp.transform_point(b.body_q[bi], wp.vec3(*ll))
            a = b.add_particle(pos=x, vel=(0, 0, 0), mass=0.0, radius=0.0, flags=0)
            b.add_spring(int(pi), a, float(k), 0.0, 0.0)
            b.spring_rest_length[-1] = 0.0
            ghost.append(a)
        seen = set()
        for mesh in b.shape_source:
            if isinstance(mesh, newton.Mesh) and id(mesh) not in seen:
                mesh.build_sdf(
                    device=self.device,
                    target_voxel_size=0.00025,
                    narrow_band_range=(-0.002, 0.002),
                    margin=0.002,
                    cache_dir=str(self.cache_dir),
                )
                seen.add(id(mesh))
        b.color()
        b.request_contact_attributes("force")
        self.model = m = b.finalize(device=self.device)
        newton.eval_fk(m, m.joint_q, m.joint_qd, m)
        m.soft_contact_ke = 2e4
        m.soft_contact_kd = 10.0
        m.soft_contact_mu = 0.15

        def pipeline(mm):
            return newton.CollisionPipeline(
                mm,
                broad_phase="sap",
                rigid_contact_max=8192,
                soft_contact_max=131072,
                soft_contact_margin=0.0005,
                enable_rigid_soft_full_surface_contact=True,
            )

        self.indent = indent
        self.vm = m
        self.rigid = SolverMuJoCo(
            m,
            use_mujoco_contacts=False,
            use_mujoco_cpu=False,
            nconmax=8192,
            njmax=65536,
            integrator="implicitfast",
        )
        all_tri = list(range(m.tri_count))
        bridge_tri = list(range(self.bridge_tri_offset, self.bridge_tri_end))
        all_edge = list(range(m.edge_count))
        bridge_edges = list(range(self.bridge_edge_offset, self.bridge_edge_end))
        vf = {i: bridge_tri for i in range(m.particle_count)}
        for i in ghost + interior + list(range(self.bridge_offset, self.real_count)):
            vf[int(i)] = all_tri
        ef = {i: bridge_edges for i in range(m.edge_count)}
        for i in bridge_edges:
            ef[i] = all_edge
        self.vbd = SolverVBD(
            m,
            iterations=iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=os.environ.get("SELF_TEST", "on") not in ("off", "soft_off"),
            particle_self_contact_margin=0.0002,
            particle_self_contact_gap=0.0001,
            particle_rest_shape_contact_exclusion_radius=0.0005,
            particle_external_vertex_contact_filtering_map=vf,
            particle_external_edge_contact_filtering_map=ef,
            rigid_body_contact_buffer_size=4096,
            rigid_body_particle_contact_buffer_size=32768,
        )
        self.feedback = wp.zeros(m.body_count, dtype=wp.spatial_vector)
        self.probe_velocity = wp.zeros(1, dtype=wp.vec3)
        self.collision = pipeline(m)
        self.raw_contacts = self.collision.contacts()
        self.contacts = self.collision.contacts()
        self.rigid_observation = self.collision.contacts()
        for key in vars(self.raw_contacts):
            if key.startswith("rigid_contact_"):
                setattr(self.contacts, key, getattr(self.raw_contacts, key))
        self.contacts.contact_generation = self.raw_contacts.contact_generation
        self.state = m.state()
        self.next = m.state()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.state)
        wp.copy(self.next.body_q, self.state.body_q)
        self.control = m.control()
        self.pi = wp.array(np.array(pis, dtype=int), dtype=int)
        self.gi = wp.array(ghost, dtype=int)
        self.bi = wp.array(bis, dtype=int)
        self.local = wp.array(
            [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in loc],
            dtype=wp.vec3,
        )
        self.k = wp.array(np.array(ks, dtype=float), dtype=float)
        self.anchor_force = wp.zeros(len(pis), dtype=wp.vec3)
        self.target = wp.clone(self.state.joint_q)
        self.caps = wp.array(
            np.array([0.8 if n.startswith("digit_") else 1.0 for n in self.joint_names], dtype=float),
            dtype=float,
        )
        self.out_f = wp.zeros(131072, dtype=wp.vec3)
        self.out_depth = wp.zeros(131072)
        self.out_normal = wp.zeros(131072, dtype=wp.vec3)
        self.out_bx = wp.zeros(131072, dtype=wp.vec3)
        self.out_reaction = wp.zeros(1, dtype=wp.vec3)
        self.action_names = [
            n for n in self.joint_names[: self.hand_dof_count] if n and n != "tissue"
        ]
        self.action_indices = np.array(
            [self.joint_names.index(n) for n in self.action_names], dtype=int
        )
        self.signs = np.array([self.convention[n] for n in self.action_names])
        self.hand_dof_indices = np.arange(self.hand_dof_count, dtype=int)
        self.hand_dof_idx = wp.array(self.hand_dof_indices, dtype=int)
        self._initial_q = self.state.joint_q.numpy().copy()
        lower = m.joint_limit_lower.numpy()[self.action_indices] * self.signs
        upper = m.joint_limit_upper.numpy()[self.action_indices] * self.signs
        self.action_lower = np.minimum(lower, upper)
        self.action_upper = np.maximum(lower, upper)
        self.region_names = ["palm", "index_proximal", "index_middle", "index_distal"]
        self.regions = np.full(self.real_count, -1, dtype=int)
        self.regions[: self.palm_count] = 0
        z = self.pads
        M = z["rest_body_matrices"][0]
        local = (z["vertices"] - M[:3, 3]) @ M[:3, :3]
        self.regions[self.pad_offset : self.bridge_offset] = np.searchsorted([0.039, 0.063], local[:, 2]) + 1
        self.body_names_all = [n.rsplit("/", 1)[-1] for n in m.body_label]
        self.region_ids = wp.array(self.regions, dtype=int)
        self.current_soft_load = wp.zeros(4)
        self.peak_soft_load = wp.zeros(4)
        self.current_rigid_load = wp.zeros(self.hand_body_count)
        self.peak_rigid_load = wp.zeros(self.hand_body_count)

    # -- brush ------------------------------------------------------------
    def _add_brush(self, b):
        spec = parse_brush_xml(self.brush_xml)
        self.brush_spec = spec
        cfg_prop = newton.ModelBuilder.ShapeConfig

        for prop in (spec["table"], spec["paper"]):
            if prop is None:
                continue
            body = b.add_link(xform=prop["xform"], mass=0.0, label=prop["name"])
            hx, hy, hz = prop["half"]
            b.add_shape_box(
                body,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=cfg_prop(
                    density=0.0,
                    mu=prop["mu"],
                    has_shape_collision=prop["collide"],
                    has_particle_collision=prop["collide"],
                ),
                label=prop["name"],
            )
            joint = b.add_joint_fixed(
                -1,
                body,
                parent_xform=prop["xform"],
                child_xform=wp.transform_identity(),
                label=f"{prop['name']}_world",
            )
            b.add_articulation([joint], label=f"{prop['name']}_articulation")
        self.prop_body_count = b.body_count - self.hand_body_count

        shaft = spec["shaft"]
        inertia = _cylinder_inertia(shaft["mass"], shaft["radius"], 2.0 * shaft["half_height"])
        root = b.add_link(
            xform=spec["root_xform"],
            com=wp.vec3(shaft["center"]),
            inertia=wp.mat33(np.diag(inertia)),
            mass=shaft["mass"],
            label="brush",
        )
        self.brush_root_body = root
        b.add_shape_cylinder(
            root,
            xform=wp.transform(wp.vec3(shaft["center"]), wp.quat_identity()),
            radius=shaft["radius"],
            half_height=shaft["half_height"],
            cfg=cfg_prop(
                density=0.0,
                mu=SHAFT_MU,
                mu_torsional=SHAFT_MU_TORSIONAL,
                mu_rolling=SHAFT_MU_ROLLING,
            ),
            label="brush_shaft",
        )
        free = b.add_joint_free(root, label="brush_free")
        joints = [free]
        stiff = {"mujoco:dof_passive_stiffness": BRUSH_HINGE_STIFFNESS}
        bristle_cfg = cfg_prop(
            density=0.0,
            mu=BRISTLE_MU,
            mu_torsional=BRISTLE_MU_TORSIONAL,
            mu_rolling=BRISTLE_MU_ROLLING,
        )
        hinge_names = []
        hinge_q = []
        hinge_dof = []
        tip_bodies = []
        root_x = spec["root_xform"]
        for segments in spec["bundles"]:
            prev_body = root
            prev_x = root_x
            for seg in segments:
                x = seg["xform"]
                half_inertia = wp.mat33(np.diag(seg["inertia"] * 0.5))
                half_mass = seg["mass"] * 0.5
                body_x = b.add_link(
                    xform=x,
                    com=seg["com"],
                    inertia=half_inertia,
                    mass=half_mass,
                    label=f"{seg['name']}_x",
                )
                jx = b.add_joint_revolute(
                    prev_body,
                    body_x,
                    parent_xform=wp.transform_multiply(wp.transform_inverse(prev_x), x),
                    child_xform=wp.transform_identity(),
                    axis=wp.vec3(1.0, 0.0, 0.0),
                    damping=BRUSH_HINGE_DAMPING,
                    target_ke=0.0,
                    target_kd=0.0,
                    armature=0.0,
                    collision_filter_parent=True,
                    custom_attributes=stiff,
                    label=f"{seg['name']}_x",
                )
                body_y = b.add_link(
                    xform=x,
                    com=seg["com"],
                    inertia=half_inertia,
                    mass=half_mass,
                    label=f"{seg['name']}_y",
                )
                jy = b.add_joint_revolute(
                    body_x,
                    body_y,
                    parent_xform=wp.transform_identity(),
                    child_xform=wp.transform_identity(),
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    damping=BRUSH_HINGE_DAMPING,
                    target_ke=0.0,
                    target_kd=0.0,
                    armature=0.0,
                    collision_filter_parent=True,
                    custom_attributes=stiff,
                    label=f"{seg['name']}_y",
                )
                b.add_shape_capsule(
                    body_y,
                    xform=wp.transform(
                        wp.vec3(0.0, 0.0, -seg["half_height"]), wp.quat_identity()
                    ),
                    radius=seg["radius"],
                    half_height=seg["half_height"],
                    cfg=bristle_cfg,
                    label=seg["name"],
                )
                for joint, name in ((jx, f"{seg['name']}_x"), (jy, f"{seg['name']}_y")):
                    joints.append(joint)
                    hinge_names.append(name)
                    hinge_dof.append(b.joint_qd_start[joint])
                    hinge_q.append(b.joint_q_start[joint])
                prev_body = body_y
                prev_x = x
            tip_bodies.append(prev_body)
        b.add_articulation(joints, label="brush_articulation")
        self.brush_joint_count = len(joints)
        self.brush_hinge_names = hinge_names
        self.brush_hinge_dof = np.array(hinge_dof, dtype=int)
        self.brush_hinge_q = np.array(hinge_q, dtype=int)
        self.brush_dof_indices = np.arange(self.hand_dof_count, len(b.joint_target_ke), dtype=int)
        self.brush_tip_bodies = tip_bodies
        self.brush_q_start = int(b.joint_q_start[free])
        self.brush_dof_start = int(b.joint_qd_start[free])

    # -- stepping ---------------------------------------------------------
    def step(self, dt=0.00025):
        self.state.clear_forces()
        wp.launch(
            anchors,
            len(self.pi),
            inputs=[
                self.state.particle_q,
                self.state.particle_qd,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.pi,
                self.gi,
                self.bi,
                self.local,
                self.k,
                self.state.body_f,
                self.anchor_force,
            ],
        )
        wp.launch(
            add_feedback, self.model.body_count, inputs=[self.state.body_f, self.feedback]
        )
        wp.launch(
            control_pd_subset,
            len(self.hand_dof_indices),
            inputs=[
                self.state.joint_q,
                self.state.joint_qd,
                self.target,
                self.caps,
                self.hand_dof_idx,
                self.control.joint_f,
            ],
        )
        self.collision.collide(self.state, self.raw_contacts)
        if os.environ.get("SELF_TEST", "on") in ("off", "rigid_off"):
            self.raw_contacts.rigid_contact_count.zero_()
        if os.environ.get("SELF_TEST", "on") in ("off", "rigidsoft_off"):
            self.raw_contacts.soft_contact_count.zero_()
        self.contacts.soft_contact_count.zero_()
        c = self.raw_contacts
        d = self.contacts
        wp.launch(
            filter_bridge_contacts,
            c.soft_contact_max,
            inputs=[
                c.soft_contact_count,
                c.soft_contact_indices,
                c.soft_contact_particle,
                c.soft_contact_barycentric,
                c.soft_contact_shape,
                c.soft_contact_body_pos,
                c.soft_contact_body_vel,
                c.soft_contact_normal,
                self.bridge_offset,
                self.real_count,
                d.soft_contact_count,
                d.soft_contact_indices,
                d.soft_contact_particle,
                d.soft_contact_barycentric,
                d.soft_contact_shape,
                d.soft_contact_body_pos,
                d.soft_contact_body_vel,
                d.soft_contact_normal,
            ],
        )
        self.rigid.step(self.state, self.next, self.control, self.contacts, dt)
        wp.launch(
            sync_ghosts,
            len(self.pi),
            inputs=[
                self.state.particle_q,
                self.state.particle_qd,
                self.next.body_q,
                self.next.body_qd,
                self.model.body_com,
                self.gi,
                self.bi,
                self.local,
            ],
        )
        self.vbd.step(self.state, self.next, self.control, self.contacts, dt)
        self.state, self.next = self.next, self.state
        self.time += dt
        self.readback_gpu(dt)
        self.feedback.zero_()
        wp.launch(
            contact_reaction,
            self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.model.shape_body,
                self.out_f,
                self.out_bx,
                self.state.body_q,
                self.model.body_com,
                self.feedback,
            ],
        )
        self.current_soft_load.zero_()
        wp.launch(
            collect_soft_load,
            self.contacts.soft_contact_max,
            inputs=[
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_indices,
                self.contacts.soft_contact_barycentric,
                self.contacts.soft_contact_shape,
                self.model.shape_body,
                self.out_f,
                self.region_ids,
                self.hand_body_count,
                self.current_soft_load,
            ],
        )
        wp.launch(record_peak, 4, inputs=[self.current_soft_load, self.peak_soft_load])
        c = self.rigid_observation
        self.rigid.update_contacts(c, self.state)
        self.current_rigid_load.zero_()
        wp.launch(
            collect_rigid_load,
            c.rigid_contact_max,
            inputs=[
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                self.model.shape_body,
                c.force,
                self.hand_body_count,
                self.current_rigid_load,
            ],
        )
        wp.launch(
            record_peak, self.hand_body_count, inputs=[self.current_rigid_load, self.peak_rigid_load]
        )

    # -- reporting --------------------------------------------------------
    def brush_info(self):
        m = self.model
        hinge = self.brush_hinge_dof
        stiff = m.mujoco.dof_passive_stiffness.numpy()
        damping = m.joint_damping.numpy()
        brush_dofs = int(m.joint_dof_count - self.hand_dof_count)
        return {
            "package_version": ax50_visualskin.__version__,
            "brush_xml": str(self.brush_xml),
            "hand_body_count": int(self.hand_body_count),
            "prop_body_count": int(self.prop_body_count),
            "brush_body_count": int(m.body_count - self.hand_body_count - self.prop_body_count),
            "total_body_count": int(m.body_count),
            "hand_dof_count": int(self.hand_dof_count),
            "brush_dof_count": brush_dofs,
            "brush_joint_count": int(self.brush_joint_count),
            "brush_hinge_count": int(len(self.brush_hinge_names)),
            "brush_hinge_dof_count": int(len(hinge)),
            "free_joint_dof": int(brush_dofs - len(hinge)),
            "brush_bundle_count": int(len(self.brush_spec["bundles"])),
            "brush_segments_per_bundle": int(len(self.brush_spec["bundles"][0])),
            "hand_q_start": 0,
            "brush_q_start": int(self.brush_q_start),
            "brush_dof_start": int(self.brush_dof_start),
            "joint_q_length": int(m.joint_q.shape[0]),
            "joint_qd_length": int(m.joint_qd.shape[0]),
            "soft_particle_count": int(m.particle_count),
            "soft_real_count": int(self.real_count),
            "hand_action_count": int(len(self.action_names)),
            "brush_stiffness_min": float(stiff[hinge].min()),
            "brush_stiffness_max": float(stiff[hinge].max()),
            "brush_damping_min": float(damping[hinge].min()),
            "brush_damping_max": float(damping[hinge].max()),
            "hand_target_ke_max": float(np.abs(m.joint_target_ke.numpy()[: self.hand_dof_count]).max()),
        }
