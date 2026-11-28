#!/usr/bin/env python3
"""Evidence probe for the AX50 pebble contact review.

Three modes run in separate processes so the legacy site-packages ``newton``
(the actual old positive-run path) and the AX50 vendored ``newton`` never share
module state:

* ``old``            - replicates scripts/nov20/env.py exactly: no
                       ``SolverMuJoCo.register_custom_attributes`` before
                       ``add_mjcf``, so the 225 ``<pair>`` entries are never
                       imported.
* ``old_registered`` - same model but with custom attributes registered before
                       ``add_mjcf``. Mechanism control only, NOT the old
                       positive path.
* ``ax50``           - AX50 ``PebbleRuntime`` (vendored newton), plus capture of
                       the ``Detected non-manifold edge`` warning.

Read-only with respect to shared sources. Writes only
``outputs/nov20/ax50_pebble07/contact_review_<mode>.json``.
"""

from __future__ import annotations

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path((str(_EVOLUTION_ROOT)))
OUT = ROOT / "outputs/nov20/ax50_pebble07"
PEBBLE_XML = ROOT / "outputs/nov20/pebble.xml"

OBJECT_GEOMS = (
    "floor", "tray", "rim_x-1", "rim_y-1", "rim_x1", "rim_y1",
    "stone0", "foot0", "stone1", "foot1", "stone2", "foot2",
)

GEOM_FIELDS = (
    "type", "friction", "solref", "solimp", "condim", "priority",
    "solmix", "margin", "gap", "contype", "conaffinity",
)
NEWTON_FIELDS = ("mu", "mu_torsional", "mu_rolling", "ke", "kd", "gap", "margin")

STEPS = 3
DT = 1.0 / 240.0


def short(name):
    return str(name).rsplit("/", 1)[-1]


def read_mj_geoms(model, names):
    out = {}
    for i in range(model.ngeom):
        name = short(model.geom(i).name)
        if name not in names or name in out:
            continue
        out[name] = {
            "geom_id": int(i),
            "type": int(model.geom_type[i]),
            "friction": [float(x) for x in model.geom_friction[i]],
            "solref": [float(x) for x in model.geom_solref[i]],
            "solimp": [float(x) for x in model.geom_solimp[i]],
            "condim": int(model.geom_condim[i]),
            "priority": int(model.geom_priority[i]),
            "solmix": float(model.geom_solmix[i]),
            "margin": float(model.geom_margin[i]),
            "gap": float(model.geom_gap[i]),
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
        }
    return out


def read_newton_shapes(model, names):
    labels = [short(x) for x in model.shape_label]
    arrays = {
        "mu": model.shape_material_mu,
        "mu_torsional": model.shape_material_mu_torsional,
        "mu_rolling": model.shape_material_mu_rolling,
        "ke": model.shape_material_ke,
        "kd": model.shape_material_kd,
        "gap": model.shape_gap,
        "margin": model.shape_margin,
    }
    arrays = {k: (v.numpy() if v is not None else None) for k, v in arrays.items()}
    out = {}
    for i, name in enumerate(labels):
        if name not in names or name in out:
            continue
        out[name] = {"shape_id": int(i)}
        for key, arr in arrays.items():
            out[name][key] = float(arr[i]) if arr is not None else None
    return out


def summarize_contacts(rows):
    groups = {}
    for row in rows:
        key = tuple(sorted((row["geom0"], row["geom1"])))
        group = groups.setdefault(key, {
            "count": 0, "friction": set(), "solref": set(), "solimp": set(),
            "dim": set(), "dist_min": row["dist"], "dist_max": row["dist"],
        })
        group["count"] += 1
        group["friction"].add(tuple(round(x, 8) for x in row["friction"]))
        group["solref"].add(tuple(round(x, 8) for x in row["solref"]))
        group["solimp"].add(tuple(round(x, 8) for x in row["solimp"]))
        group["dim"].add(int(row["dim"]))
        group["dist_min"] = min(group["dist_min"], row["dist"])
        group["dist_max"] = max(group["dist_max"], row["dist"])
    out = {}
    for key, group in sorted(groups.items()):
        out["|".join(key)] = {
            "count": group["count"],
            "friction": sorted(group["friction"]),
            "solref": sorted(group["solref"]),
            "solimp": sorted(group["solimp"]),
            "dim": sorted(group["dim"]),
            "dist_min": group["dist_min"],
            "dist_max": group["dist_max"],
        }
    return out


def old_contacts(solver):
    data = solver.mj_data
    model = solver.mj_model
    names = [short(model.geom(i).name) for i in range(model.ngeom)]
    rows = []
    for ci in range(int(data.ncon)):
        contact = data.contact[ci]
        g0, g1 = int(contact.geom[0]), int(contact.geom[1])
        rows.append({
            "geom0": names[g0],
            "geom1": names[g1],
            "friction": [float(x) for x in contact.friction],
            "solref": [float(x) for x in contact.solref],
            "solimp": [float(x) for x in contact.solimp],
            "dim": int(contact.dim),
            "dist": float(contact.dist),
        })
    return rows


def ax50_contacts(h):
    model = h.rigid.mj_model
    data = h.rigid.mjw_data
    names = [short(model.geom(i).name) for i in range(model.ngeom)]
    nacon = np.asarray(data.nacon.numpy()).reshape(-1)
    n = int(nacon[0]) if nacon.size else 0
    contact = data.contact

    def take(field):
        return getattr(contact, field).numpy()[:n]

    geom = take("geom")
    friction = take("friction")
    solref = take("solref")
    solimp = take("solimp")
    dim = take("dim")
    dist = take("dist")
    rows = []
    for i in range(n):
        g0, g1 = int(geom[i, 0]), int(geom[i, 1])
        rows.append({
            "geom0": names[g0],
            "geom1": names[g1],
            "friction": [float(x) for x in friction[i]],
            "solref": [float(x) for x in solref[i]],
            "solimp": [float(x) for x in solimp[i]],
            "dim": int(dim[i]),
            "dist": float(dist[i]),
        })
    return rows, nacon.tolist()


def pair_rows(model):
    count = int(model.custom_frequency_counts.get("mujoco:pair", 0))
    if count == 0:
        return []
    attrs = getattr(model, "mujoco", None)
    if attrs is None:
        return []

    def get(name):
        value = getattr(attrs, name, None)
        return value.numpy() if value is not None else None

    geom1 = get("pair_geom1")
    geom2 = get("pair_geom2")
    if geom1 is None or geom2 is None:
        return []
    friction = get("pair_friction")
    solref = get("pair_solref")
    labels = [short(x) for x in model.shape_label]
    rows = []
    for i in range(count):
        row = {
            "shape1": int(geom1[i]),
            "shape2": int(geom2[i]),
            "geom1": labels[int(geom1[i])],
            "geom2": labels[int(geom2[i])],
        }
        if friction is not None:
            row["friction"] = [float(x) for x in friction[i]]
        if solref is not None:
            row["solref"] = [float(x) for x in solref[i]]
        rows.append(row)
    return rows


def run_old(register_pairs):
    import mujoco  # noqa: F401  (same import env.py relies on)
    import newton

    builder = newton.ModelBuilder()
    builder.default_joint_cfg.target_ke = 2.0
    builder.default_joint_cfg.target_kd = 0.08
    builder.default_joint_cfg.armature = 0.0003
    if register_pairs:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(str(PEBBLE_XML), collapse_fixed_joints=False)

    joints = [short(x) for x in builder.joint_label]
    for i, name in enumerate(joints):
        k = builder.joint_qd_start[i]
        if "diagnostic_wrist" in name:
            builder.joint_target_ke[k] = 1700.0
            builder.joint_target_kd[k] = 42.0
        if name == "hinge":
            builder.joint_target_ke[k] = 0.0
            builder.joint_target_kd[k] = 0.0

    model = builder.finalize(device="cpu")
    state = model.state()
    next_state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=True, nconmax=4096, njmax=2048, iterations=40)

    for _ in range(STEPS):
        state.clear_forces()
        solver.step(state, next_state, control, None, DT)
        state, next_state = next_state, state

    contacts = old_contacts(solver)
    return {
        "mode": "old_registered" if register_pairs else "old",
        "newton_version": str(newton.__version__),
        "newton_path": str(newton.__file__),
        "xml": str(PEBBLE_XML),
        "register_pairs": bool(register_pairs),
        "npair": int(solver.mj_model.npair),
        "ngeom": int(solver.mj_model.ngeom),
        "custom_frequency_counts": dict(model.custom_frequency_counts),
        "pairs": pair_rows(model),
        "use_mujoco_contacts": bool(solver._use_mujoco_contacts),
        "steps": STEPS,
        "dt": DT,
        "mj_geoms": read_mj_geoms(solver.mj_model, OBJECT_GEOMS),
        "newton_shapes": read_newton_shapes(model, OBJECT_GEOMS),
        "contacts": {
            "ncon": len(contacts),
            "rows": contacts,
            "summary": summarize_contacts(contacts),
        },
    }


def pads_npz_check(asset, pad_offset):
    """Independent numpy check of the pads tet-mesh boundary surface."""
    z = np.load(Path(asset) / "pads.npz")
    tets = np.asarray(z["tets"], dtype=np.int64)
    faces = np.concatenate(
        (tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]),
        axis=0)
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    boundary = np.unique(np.sort(faces, axis=1), axis=0)[counts == 1]
    edges = np.sort(np.concatenate(
        (boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]])), axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    bad = unique_edges[edge_counts > 2]
    return {
        "asset": str(Path(asset) / "pads.npz"),
        "tets": int(len(tets)),
        "boundary_triangles": int(len(boundary)),
        "unique_edges": int(len(unique_edges)),
        "nonmanifold_edges_local": bad.tolist(),
        "nonmanifold_edges_global": (bad + int(pad_offset)).tolist(),
        "incidence": [int(x) for x in edge_counts[edge_counts > 2]],
    }


def topology_check(h, mesh_info):
    """Measured: are the non-manifold pads triangles/edges still in the model?"""
    model = h.model
    tri = model.tri_indices.numpy()
    edge = model.edge_indices.numpy()
    adj = model.soft_mesh_adjacency

    pads_tri_n = int(mesh_info["triangles"])
    tri_end = int(h.bridge_tri_offset)
    tri_start = tri_end - pads_tri_n
    pads_tris = tri[tri_start:tri_end]

    pads_edge_n = int(mesh_info["edge_count"])
    edge_end = int(h.bridge_edge_offset)
    edge_start = edge_end - pads_edge_n
    pads_edges = edge[edge_start:edge_end]

    tri_edges = np.sort(np.concatenate(
        (pads_tris[:, [0, 1]], pads_tris[:, [1, 2]], pads_tris[:, [2, 0]])), axis=1)
    unique_tri_edges = np.unique(tri_edges, axis=0)

    bad_edges = np.asarray(mesh_info["nonmanifold_edges"], dtype=np.int64)
    present = []
    linked = []
    for bad in bad_edges:
        key = np.sort(bad)
        hit = np.where(np.all(np.sort(pads_edges[:, 2:4], axis=1) == key, axis=1))[0]
        row = int(hit[0]) + edge_start if hit.size else None
        present.append(bool(hit.size))
        if row is None:
            linked.append(None)
            continue
        incident = np.where(
            ((pads_tris == key[0]).any(axis=1)) & ((pads_tris == key[1]).any(axis=1)))[0]
        n_linked = int((adj.tri_edge_indices[np.asarray(incident) + tri_start] == row).any(axis=1).sum())
        linked.append({"incident_triangles": int(incident.size), "triangles_linked_to_edge": n_linked})

    v_adj = np.asarray(adj.v_adj_edges)
    v_off = np.asarray(adj.v_adj_edges_offsets)
    adjacency_has_edges = []
    for bad in bad_edges:
        ok = True
        for a, b in ((bad[0], bad[1]), (bad[1], bad[0])):
            neighbors = v_adj[v_off[a]:v_off[a + 1]]
            edge_ids = np.flatnonzero(np.all(np.sort(edge[:, 2:4], axis=1) == np.sort([a, b]), axis=1))
            ok = ok and bool(np.isin(neighbors, edge_ids).any())
        adjacency_has_edges.append(ok)

    return {
        "pads_tri_range": [tri_start, tri_end],
        "pads_triangles_in_model": int(len(pads_tris)),
        "pads_triangle_vertex_min": int(pads_tris.min()),
        "pads_triangle_vertex_max": int(pads_tris.max()),
        "pads_edge_range": [edge_start, edge_end],
        "pads_edges_in_model": int(len(pads_edges)),
        "pads_unique_edges_from_model_triangles": int(len(unique_tri_edges)),
        "nonmanifold_edge_present_in_model": present,
        "nonmanifold_edge_vertex_adjacency": adjacency_has_edges,
        "nonmanifold_edge_triangle_links": linked,
        "note": ("edge_indices keeps the edge endpoints for every unique edge; "
                 "tri_edge_indices links at most one triangle per side, so the "
                 "3rd/4th incident triangles stay unlinked"),
    }


def run_ax50():
    sys.path.insert(0, str(ROOT / "scripts/nov20/ax50_pebble07"))
    from pebble_runtime import PebbleRuntime
    from newton._src.utils.mesh import MeshAdjacency

    captured = []
    original_show = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        captured.append({
            "message": str(message),
            "category": category.__name__,
            "filename": str(filename),
            "lineno": int(lineno),
            "stack": traceback.format_stack(limit=20),
        })

    warnings.showwarning = showwarning
    original_init = MeshAdjacency.__init__

    def init(self, *args, **kwargs):
        before = len(captured)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            original_init(self, *args, **kwargs)
        new = captured[before:]
        if new:
            tri = np.asarray(self.indices).reshape(-1, 3)
            edges = np.sort(np.concatenate(
                (tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
            unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
            info = {
                "triangles": int(len(tri)),
                "unique_vertices": int(len(np.unique(tri))),
                "vertex_min": int(tri.min()),
                "vertex_max": int(tri.max()),
                "edge_count": int(len(unique_edges)),
                "nonmanifold_edge_count": int((counts > 2).sum()),
                "nonmanifold_edges": unique_edges[counts > 2].tolist(),
                "incidence": [int(x) for x in counts[counts > 2]],
            }
            for rec in new:
                rec["mesh"] = info

    MeshAdjacency.__init__ = init
    try:
        h = PebbleRuntime(side="left", wrist="six_axis", position=(.25, .25, .35),
                          gravity=(0, 0, -9.81))
        build_capture = list(captured)
        h.advance(.01, dt=.00025)
        advance_capture = captured[len(build_capture):]
    finally:
        MeshAdjacency.__init__ = original_init
        warnings.showwarning = original_show

    contacts, nacon_raw = ax50_contacts(h)
    model = h.model
    mj_model = h.rigid.mj_model

    mesh_info = None
    for rec in build_capture:
        if "mesh" in rec:
            mesh_info = rec["mesh"]
            break

    nonmanifold = {
        "captured_warnings": build_capture,
        "warnings_during_advance": advance_capture,
        "mesh": mesh_info,
        "pad_offset": int(h.pad_offset),
        "palm_count": int(h.palm_count),
        "bridge_offset": int(h.bridge_offset),
        "bridge_tri_offset": int(h.bridge_tri_offset),
        "bridge_edge_offset": int(h.bridge_edge_offset),
    }
    if mesh_info is not None:
        if h.pad_offset <= mesh_info["vertex_min"] and mesh_info["vertex_max"] < h.bridge_offset:
            nonmanifold["source_mesh"] = "pads"
        elif mesh_info["vertex_max"] < h.pad_offset:
            nonmanifold["source_mesh"] = "palm"
        else:
            nonmanifold["source_mesh"] = "bridges_or_unknown"
        nonmanifold["pads_npz_check"] = pads_npz_check(h.asset, h.pad_offset)
        nonmanifold["topology_check"] = topology_check(h, mesh_info)

    return {
        "mode": "ax50",
        "newton_version": str(__import__("newton").__version__),
        "newton_path": str(__import__("newton").__file__),
        "xml": str(h.object_xml),
        "npair": int(mj_model.npair),
        "ngeom": int(mj_model.ngeom),
        "custom_frequency_counts": dict(model.custom_frequency_counts),
        "pairs": pair_rows(model),
        "use_mujoco_contacts": bool(h.rigid._use_mujoco_contacts),
        "per_contact_shape_properties": bool(
            getattr(h.contacts, "per_contact_shape_properties", False)),
        "steps": 40,
        "dt": .00025,
        "mj_geoms": read_mj_geoms(mj_model, OBJECT_GEOMS),
        "newton_shapes": read_newton_shapes(model, OBJECT_GEOMS),
        "contacts": {
            "nacon_raw": nacon_raw,
            "ncon": len(contacts),
            "rows": contacts,
            "summary": summarize_contacts(contacts),
        },
        "nonmanifold": nonmanifold,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=("old", "old_registered", "ax50"))
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "ax50":
        result = run_ax50()
    else:
        result = run_old(args.mode == "old_registered")

    path = OUT / f"contact_review_{args.mode}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        "mode": result["mode"],
        "newton": result["newton_version"],
        "npair": result["npair"],
        "ngeom": result["ngeom"],
        "ncon": result["contacts"]["ncon"],
        "per_contact_shape_properties": result.get("per_contact_shape_properties"),
    }
    if result["mode"] == "ax50":
        nm = result["nonmanifold"]
        summary["nonmanifold_source"] = nm.get("source_mesh")
        summary["nonmanifold_warnings"] = len(nm["captured_warnings"])
    print("CONTACT_REVIEW_PROBE", json.dumps(summary), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
