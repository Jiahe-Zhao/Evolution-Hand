"""Morphology and generated collision-geometry gates.

The source gate accepts legal topology changes, while the generated-URDF gate
rejects collision assets that cannot be loaded as closed physical solids.
"""
import math
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def _topology_module():
    path = Path(__file__).resolve().parents[1] / "evolution_tasks" / "collision_topology.py"
    if not path.is_file():
        path = Path(__file__).with_name("collision_topology.py")
    spec = importlib.util.spec_from_file_location("hand_collision_topology", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _origin(node):
    value = np.eye(4)
    if node is not None:
        xyz = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError("Invalid or nonfinite origin")
        value[:3, 3] = xyz
        value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return value


def _interior_depth(mesh, point):
    """Signed distance at an explicit witness, without convexifying the solid."""
    triangles = mesh.triangles
    points = np.broadcast_to(point, (len(triangles), 3))
    closest = trimesh.triangles.closest_point(triangles, points)
    distance = float(np.linalg.norm(closest - point, axis=1).min())
    a, b, c = (triangles[:, i] - point for i in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la + np.einsum("ij,ij->i", c, a) * lb
    inside = abs(np.sum(2 * np.arctan2(numerator, denominator))) > 2 * np.pi
    return distance if inside else -distance


def _audit_neutral_overlap(path, root, local_meshes, world_poses=None):
    """Reject only witnessed deep overlap of non-neighbour physical solids."""
    topology = _topology_module()
    links, parents, _ = topology.collision_graph(path)
    bodies = {}
    for name in links:
        body = name
        while body in parents and parents[body][1] == "fixed":
            body = parents[body][0]
        bodies[name] = body
    excluded = set(topology.collision_exclusions(path, set(bodies.values())))
    poses = world_poses if world_poses is not None else {name: np.eye(4) for name in links if name not in parents}
    pending = [] if world_poses is not None else list(root.findall("joint"))
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in poses]
        if not ready:
            raise ValueError("Disconnected or cyclic URDF")
        for joint in ready:
            transform = _origin(joint.find("origin"))
            if joint.get("type") != "fixed":
                limit = joint.find("limit")
                q = float(np.clip(0.0, float(limit.get("lower")), float(limit.get("upper"))))
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion = np.eye(4)
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = axis / np.linalg.norm(axis) * q
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * q).as_matrix()
                transform = transform @ motion
            poses[joint.find("child").get("link")] = poses[joint.find("parent").get("link")] @ transform
            pending.remove(joint)
    meshes = []
    thicknesses = {}
    for name, index, mesh, transform in local_meshes:
        thicknesses[(name, index)] = float(mesh.extents.min())
        world = mesh.copy()
        world.apply_transform(poses[name] @ transform)
        meshes.append((name, index, world, ConvexHull(world.vertices).equations))
    records, tested, filtered = [], 0, 0
    for i, (a, ai, ma, ea) in enumerate(meshes):
        for b, bi, mb, eb in meshes[i + 1:]:
            pair = tuple(sorted((bodies[a], bodies[b])))
            if bodies[a] == bodies[b] or pair in excluded:
                filtered += 1
                continue
            tested += 1
            low, high = np.maximum(ma.bounds[0], mb.bounds[0]), np.minimum(ma.bounds[1], mb.bounds[1])
            if np.any(high <= low):
                continue
            equations = np.vstack((ea, eb))
            solution = linprog([0, 0, 0, -1], A_ub=np.c_[equations[:, :3], np.ones(len(equations))],
                               b_ub=-equations[:, 3], bounds=[(None, None)] * 3 + [(0, None)], method="highs")
            if not solution.success or solution.x[3] <= 1e-6:
                continue
            witness = solution.x[:3]
            depth = min(_interior_depth(ma, witness), _interior_depth(mb, witness))
            tolerance = max(0.0005, 0.15 * min(thicknesses[(a, ai)], thicknesses[(b, bi)]))
            records.append({
                "a": f"{a}:{ai}", "b": f"{b}:{bi}",
                "convex_overlap_radius_m": float(solution.x[3]),
                "solid_witness_depth_m": float(depth), "severe_tolerance_m": float(tolerance),
                "witness_m": witness.tolist(), "severe": bool(depth > tolerance),
            })
    return {
        "scope": "All non-neighbour pairs at zero/clamped joint pose; convex broad phase, closed-triangle interior witness. Not a proof over the continuous ROM.",
        "tested_pairs": tested, "filtered_pairs": filtered, "overlaps": records,
        "passed": not any(r["severe"] for r in records),
    }


def validate_morphology(urdf_info):
    reasons = []
    links = list(urdf_info.get("links", []))
    bases = list(urdf_info.get("base_link", []))
    entries = bases + links
    names = [link.get("name_code") for link in entries]
    if not bases or any(not isinstance(name, str) or not name for name in names):
        reasons.append("missing base or link name")
    if len(set(names)) != len(names):
        reasons.append("duplicate link names")
    by_name = {link.get("name_code"): link for link in entries}
    base_names = {link.get("name_code") for link in bases}

    def finite(value):
        try:
            return math.isfinite(float(value))
        except (ValueError, TypeError):
            return False

    def vector(value):
        return isinstance(value, (tuple, list)) and len(value) == 3 and all(finite(x) for x in value)

    for link in entries:
        name = link.get("name_code")
        kind = link.get("geometry_type")
        dimensions = (link.get("geometry_size") if kind == "box" else
                      [link.get("geometry_radius"), link.get("geometry_length")])
        if kind not in {"box", "capsule", "cylinder"}:
            reasons.append(f"unsupported source geometry requiring manual audit: {name}/{kind}")
        elif not dimensions or (kind == "box" and len(dimensions) != 3) or any(
            not finite(x) or float(x) <= 0 for x in dimensions
        ):
            reasons.append(f"nonpositive or nonfinite geometry: {name}")
        if name in base_names:
            continue
        for field in ("joint_origin_translation", "joint_origin_rpy"):
            if not vector(link.get(field)):
                reasons.append(f"invalid {field}: {name}")
        if link.get("joint_type") != "fixed":
            axis = link.get("joint_axis")
            if not vector(axis) or sum(float(x) ** 2 for x in axis) < 1e-12:
                reasons.append(f"invalid joint axis: {name}")
            limits = link.get("joint_limit") or {}
            lower, upper = limits.get("lower"), limits.get("upper")
            if not finite(lower) or not finite(upper) or float(lower) >= float(upper):
                reasons.append(f"invalid joint limits: {name}")
        visited = set()
        current = name
        while current not in base_names:
            if current in visited:
                reasons.append(f"joint cycle at {name}")
                break
            visited.add(current)
            if current not in by_name:
                reasons.append(f"missing parent {current} for {name}")
                break
            current = by_name[current].get("joint_parent")

    reasons = sorted(set(reasons))
    return not reasons, {
        "passed": not reasons, "gate": "source_topology_v2", "reasons": reasons,
        "warnings": [], "link_count": len(links), "collision_verified": False,
        "collision_check_scope": "Not checked here; inspect generated URDF and PhysX geometry separately.",
    }


def lightweight_geometry_prefilter(urdf_info):
    """Cheap CPU-only filter before any Isaac/PhysX process is started.

    This rejects only malformed or numerically unsafe mutations.  It does not
    reject ordinary self-contact or a pose-dependent overlap; those remain the
    responsibility of the full runtime gate after this filter passes.
    """
    passed, report = validate_morphology(urdf_info)
    reasons = list(report.get("reasons", []))
    entries = list(urdf_info.get("base_link", [])) + list(urdf_info.get("links", []))
    by_name = {link.get("name_code"): link for link in entries}

    def parse_digit(name):
        if not isinstance(name, str) or not name.startswith("link_"):
            return None
        parts = name.split("_")
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            return None
        return int(parts[1]), int(parts[2])

    # A generated phalanx must start at its parent's distal end.  This is a
    # cheap deterministic check and prevents launching Isaac for candidates
    # whose source kinematics are already inconsistent.
    digit_segments = {}
    for link in entries:
        name = link.get("name_code")
        code = parse_digit(name)
        if code:
            digit_segments.setdefault(code[0], []).append(code[1])
        parent_name = link.get("joint_parent")
        parent_code = parse_digit(parent_name)
        if not code or not parent_code or code[0] != parent_code[0] or parent_name not in by_name:
            continue
        if code[1] != parent_code[1] + 1:
            continue
        parent = by_name[parent_name]
        try:
            expected_z = float(parent.get("geometry_length")) + float(parent.get("geometry_radius"))
            actual_z = float((link.get("joint_origin_translation") or [0, 0, 0])[2])
        except (TypeError, ValueError, IndexError):
            reasons.append(f"invalid_kinematic_connection:{parent_name}->{name}")
            continue
        tolerance = max(0.0015, 0.08 * expected_z)
        if abs(actual_z - expected_z) > tolerance:
            reasons.append(
                f"unsynchronized_child_origin:{parent_name}->{name}:"
                f"actual={actual_z:.6f},expected={expected_z:.6f}"
            )
    for finger_id, segments in digit_segments.items():
        ordered = sorted(set(segments))
        if ordered and ordered != list(range(ordered[-1] + 1)):
            reasons.append(f"noncontiguous_digit_chain:{finger_id}:{ordered}")

    # Conservative CPU broad phase for neighboring finger roots.  The full
    # PhysX gate remains authoritative, but this catches obvious root-level
    # interpenetration without starting Isaac.
    roots = []
    for link in entries:
        code = parse_digit(link.get("name_code"))
        if not code or code[1] != 0:
            continue
        t = link.get("joint_origin_translation") or [0.0, 0.0, 0.0]
        try:
            center = [float(t[0]), float(t[1]), float(t[2])]
            radius = float(link.get("geometry_radius", 0.0))
        except (TypeError, ValueError, IndexError):
            continue
        roots.append((code[0], center, radius))
    for index, (finger_a, center_a, radius_a) in enumerate(roots):
        for finger_b, center_b, radius_b in roots[index + 1:]:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(center_a, center_b)))
            # Leave a small allowance for the calibrated MCP fan; only reject
            # clear overlap beyond that allowance.
            if distance + 0.001 < radius_a + radius_b:
                reasons.append(
                    f"root_bbox_overlap:{finger_a}:{finger_b}:"
                    f"distance={distance:.6f},limit={radius_a + radius_b - 0.001:.6f}"
                )
    for link in entries:
        name = link.get("name_code", "<unnamed>")
        kind = link.get("geometry_type")
        if kind == "box":
            dimensions = link.get("geometry_size") or []
        else:
            dimensions = [link.get("geometry_radius"), link.get("geometry_length")]
        try:
            values = [float(value) for value in dimensions]
        except (TypeError, ValueError):
            values = []
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            reasons.append(f"unsafe_geometry:{name}")
            continue
        # Keep mutation outputs physically meaningful without imposing the
        # stricter continuous-ROM collision policy used by Isaac.
        if kind != "box" and (values[0] < 0.0005 or values[0] > 0.05):
            reasons.append(f"radius_out_of_prefilter_range:{name}")
        if kind != "box" and (values[1] < 0.003 or values[1] > 0.25):
            reasons.append(f"length_out_of_prefilter_range:{name}")
        if kind == "box" and any(value < 0.001 or value > 0.30 for value in values):
            reasons.append(f"box_extent_out_of_prefilter_range:{name}")
        for field in ("joint_origin_translation", "joint_origin_rpy"):
            vector = link.get(field)
            if vector is not None:
                try:
                    if any(not math.isfinite(float(value)) or abs(float(value)) > 10.0 for value in vector):
                        reasons.append(f"unsafe_transform:{name}:{field}")
                except (TypeError, ValueError):
                    reasons.append(f"unsafe_transform:{name}:{field}")
    reasons = sorted(set(reasons))
    return not reasons, {
        "passed": not reasons,
        "gate": "lightweight_geometry_prefilter_v1",
        "reasons": reasons,
        "delegates_runtime_collision": True,
    }


def audit_generated_urdf(urdf_path):
    """Validate every generated collision mesh before it reaches PhysX.

    This intentionally does not reject a hand merely because a neutral pose
    self-intersects: a closed hand can have legitimate adjacent contact. It
    rejects malformed geometry, missing assets, invalid transforms, and broken
    URDF references, which are the failures that cannot be adapted safely.
    """
    path = Path(urdf_path)
    reasons = []
    records = []
    local_meshes = []
    if not path.is_file():
        return False, {"passed": False, "gate": "generated_urdf_v1", "reasons": ["missing_urdf"]}
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return False, {"passed": False, "gate": "generated_urdf_v1", "reasons": [f"urdf_parse:{exc}"]}

    link_names = {link.get("name") for link in root.findall("link")}
    if None in link_names or len(link_names) != len(root.findall("link")):
        reasons.append("missing_or_duplicate_link_name")
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None or parent.get("link") not in link_names or child.get("link") not in link_names:
            reasons.append(f"broken_joint:{joint.get('name', '<unnamed>')}")
        try:
            _origin(joint.find("origin"))
            if joint.get("type") not in {"fixed", "revolute", "prismatic"}:
                raise ValueError("Unsupported/unbounded joint type")
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                limit = joint.find("limit")
                lower, upper, effort, velocity = [float(limit.get(k)) for k in ("lower", "upper", "effort", "velocity")]
                if axis.size != 3 or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8 or not np.isfinite([lower, upper, effort, velocity]).all() or lower >= upper or effort <= 0 or velocity <= 0:
                    raise ValueError("Invalid axis or limits")
        except (TypeError, ValueError, AttributeError) as exc:
            reasons.append(f"joint_parameters:{joint.get('name')}:{exc}")

    for link in root.findall("link"):
        if link.findall("visual") and not link.findall("collision"):
            reasons.append(f"visible_link_without_collision:{link.get('name')}")
        inertial = link.find("inertial")
        try:
            _origin(inertial.find("origin"))
            mass = float(inertial.find("mass").get("value"))
            values = inertial.find("inertia").attrib
            inertia = np.array([[float(values["ixx"]), float(values["ixy"]), float(values["ixz"])],
                                [float(values["ixy"]), float(values["iyy"]), float(values["iyz"])],
                                [float(values["ixz"]), float(values["iyz"]), float(values["izz"])]])
            if not np.isfinite(mass) or mass <= 0 or not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0:
                raise ValueError("Mass/inertia must be finite and positive definite")
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            reasons.append(f"invalid_inertia:{link.get('name')}:{exc}")
        for index, collision in enumerate(link.findall("collision")):
            mesh_node = collision.find("geometry/mesh")
            if mesh_node is None or not mesh_node.get("filename"):
                reasons.append(f"missing_collision_mesh:{link.get('name')}:{index}")
                continue
            mesh_path = Path(mesh_node.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = path.parent / mesh_path
            record = {"link": link.get("name"), "index": index, "path": str(mesh_path)}
            try:
                # STL stores independent triangle vertices; weld duplicates
                # before checking whether edges bound a closed solid.
                mesh = trimesh.load(mesh_path, force="mesh", process=True)
                scale = np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" ")
                if scale.size != 3 or not np.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError("invalid_mesh_scale")
                mesh.apply_scale(scale)
                transform = _origin(collision.find("origin"))
                record.update({
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                    "winding_consistent": bool(mesh.is_winding_consistent),
                    "finite": bool(np.isfinite(mesh.vertices).all()),
                    "volume": float(mesh.volume),
                })
                if not record["finite"] or not record["watertight"] or not record["winding_consistent"] or record["volume"] <= 1e-12:
                    reasons.append(f"invalid_collision_solid:{link.get('name')}:{index}")
                else:
                    local_meshes.append((link.get("name"), index, mesh, transform))
            except Exception as exc:  # mesh loaders raise several backend-specific types
                record["error"] = str(exc)
                reasons.append(f"collision_mesh_load:{link.get('name')}:{index}")
            records.append(record)

    if not records:
        reasons.append("no_collision_meshes")
    neutral_report = None
    fingerprint = None
    if not reasons:
        try:
            neutral_report = _audit_neutral_overlap(path, root, local_meshes)
            if not neutral_report["passed"]:
                reasons.append("severe_nonadjacent_neutral_interpenetration")
            fingerprint = _topology_module().collision_asset_fingerprint(path)
        except Exception as exc:
            reasons.append(f"topology_or_neutral_audit:{exc}")
    reasons = sorted(set(reasons))
    return not reasons, {
        "passed": not reasons,
        "gate": "generated_urdf_v2",
        "reasons": reasons,
        "collision_mesh_count": len(records),
        "collision_meshes": records,
        "asset_fingerprint": fingerprint,
        "neutral_self_collision": neutral_report,
        "collision_verified": False,
        "mesh_integrity_verified": not reasons,
        "remaining_checks": ["dynamic_self_collision_ROM", "PhysX_cooked_geometry", "task_object_placement"],
    }
