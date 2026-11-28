"""AX50 v3 hand + cleaned fruit-02 free fruit with tension-only breakable stem."""

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

import ax50_visualskin  # noqa: F401,E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

import damped_adapter  # noqa: E402
import pair_damping  # noqa: E402

SRC_XML = ROOT / "outputs/nov20/fruit02/pick.xml"
OUT_DIR = ROOT / "outputs/nov20/ax50_fruit02"
CLEAN_XML = OUT_DIR / "fruit_clean.xml"

STEM_ANCHOR = wp.vec3(-0.093, 0.003, 0.127)
STEM_TIP_LOCAL = wp.vec3(0.0, 0.0, 0.014)
STEM_L0 = 0.030
STEM_K = 55.0
STEM_C = 0.18
STEM_BREAK_N = 0.45


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
        kp = 1500.0
        kd = 30.0
        cap = 15.0
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


@wp.kernel
def stem_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_f: wp.array(dtype=wp.spatial_vector),
    fruit_idx: int,
    dt: float,
    stem_state: wp.array(dtype=float),  # [0]=attached(1/0), [1]=prev_len, [2]=last_demand_N, [3]=break_event(1/0), [4]=peak_demand_N
):
    attached = stem_state[0]
    prev_len = stem_state[1]
    tf = body_q[fruit_idx]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    off = wp.quat_rotate(q, STEM_TIP_LOCAL)
    pos = p + off
    delta = STEM_ANCHOR - pos
    length = wp.length(delta)
    F = float(0.0)
    if attached > 0.5 and length > STEM_L0:
        dlen_dt = (length - prev_len) / dt
        F = wp.max(0.0, STEM_K * (length - STEM_L0) + STEM_C * dlen_dt)
    stem_state[1] = length
    stem_state[2] = F
    if F > stem_state[4]:
        stem_state[4] = F
    if attached > 0.5 and F >= STEM_BREAK_N:
        stem_state[0] = 0.0
        stem_state[3] = 1.0
        attached = 0.0
    else:
        stem_state[3] = 0.0
    if attached > 0.5 and F > 0.0 and length > 1.0e-8:
        f_vec = (F / length) * delta
        tau = wp.cross(off, f_vec)
        body_f[fruit_idx] = body_f[fruit_idx] + wp.spatial_vector(f_vec, tau)


class FruitVBD(pair_damping.KeyPairVBD):
    """KeyPairVBD plus fruit-02 soft-contact damping override."""

    def _refresh_body_particle_contact_state(self, contacts, refresh):
        super()._refresh_body_particle_contact_state(contacts, refresh)
        for shape in getattr(self, "extra_palm_shapes", []):
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
        flags = getattr(self, "prop_shape_flags", None)
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
            override_ke = getattr(self, "prop_contact_ke", None)
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


pair_damping.KeyPairVBD = FruitVBD


def clean_fruit_xml(src=SRC_XML, dst=CLEAN_XML):
    """Strip legacy hand_111 subtree/actuators/contact pairs, keep branch + fruit."""
    src = Path(src)
    dst = Path(dst)
    tree = _evo_parse_xml(src)
    root = tree.getroot()

    act = root.find("actuator")
    if act is not None:
        root.remove(act)

    world = root.find("worldbody")
    diag = world.find("./body[@name=\"diagnostic_world\"]")
    if diag is not None:
        world.remove(diag)

    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh" and child.attrib.get("name") != "fruitSkin":
                asset.remove(child)

    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)

    ET.indent(tree, space=" ")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return {
        "source": str(src),
        "clean": str(dst),
        "kept_bodies": [b.attrib.get("name") for b in world.iter("body")],
        "kept_world_geoms": [c.attrib.get("name") for c in world if c.tag == "geom"],
    }


class FruitRuntime(damped_adapter.AX50BrushRuntime):
    """AX50 hand + free fruit with tension-only breakable stem."""

    def __init__(self, *args, fruit_xml=None, **kwargs):
        self.source_xml = Path(fruit_xml or SRC_XML)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.clean_xml = CLEAN_XML
        self.clean_summary = clean_fruit_xml(self.source_xml, self.clean_xml)
        super().__init__(*args, **kwargs)
        self.stem_state = wp.array(
            np.array([1.0, STEM_L0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=float
        )
        self.break_happened = False
        self.break_time_s = None
        self.break_demand_N = 0.0

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

        labels = [str(x) for x in b.body_label]
        self.fruit_body = next(i for i, n in enumerate(labels) if n.rsplit("/", 1)[-1] == "fruit")
        self.prop_body_count = int(b.body_count - self.hand_body_count)

    def _build(self, iterations, indent, gravity, scene_builder):
        super()._build(iterations, indent, gravity, scene_builder)
        m = self.model
        shape_body = m.shape_body.numpy()
        flags = np.array(
            [1 if (int(b) == self.fruit_body or int(b) < 0) else 0 for b in shape_body],
            dtype=int,
        )
        self.prop_shape_flags_np = flags
        self.vbd.extra_palm_shapes = [
            i for i, label in enumerate(m.shape_label)
            if str(label).endswith(("/link_2_0/palm_tile_12", "/link_2_0/palm_tile_14"))
        ]
        self.vbd.prop_shape_flags = wp.array(flags, dtype=int)
        self.vbd.prop_contact_kd = float(os.environ.get("FRUIT_KD", "0.2"))
        # Align fruit friction coefficient to original fruit02.py (0.85)
        mu = m.shape_material_mu.numpy()
        for idx_s, b_id in enumerate(shape_body):
            if int(b_id) == self.fruit_body:
                mu[idx_s] = 0.85
        m.shape_material_mu.assign(mu)

    def step(self, dt=0.000125):
        st = self.state
        orig_clear = st.clear_forces
        def clear_and_stem():
            orig_clear()
            wp.launch(
                stem_force_kernel,
                1,
                inputs=[st.body_q, st.body_f, self.fruit_body, float(dt), self.stem_state],
            )
        st.clear_forces = clear_and_stem
        try:
            super().step(dt)
        finally:
            st.clear_forces = orig_clear
