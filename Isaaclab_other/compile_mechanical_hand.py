"""Map evolved bone URDF geometry to rigid link, bearing and fingertip modules.

Physical masses and inertias are computed from manufactured-part proxies.
Actuation is a joint-side servo abstraction; transmission packaging is recorded
in the manifest and is not a simulated cable or a certified manufacturing CAD.
"""
import argparse
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return t


def palm_chassis(root, source, parent_link, original_root):
    roots = []
    long_roots = []
    radii = []
    original_links = {link.get("name"): link for link in original_root.findall("link")}
    joints = root.findall("joint")
    parents = {joint.find("child").get("link"): joint for joint in joints}
    poses = {parent_link: np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [joint for joint in pending if joint.find("parent").get("link") in poses]
        if not ready:
            raise ValueError("Cannot resolve bone hand kinematics for palm generation")
        for joint in ready:
            transform = origin(joint.find("origin"))
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(
                    axis * float(np.clip(0., float(joint.find("limit").get("lower")),
                                         float(joint.find("limit").get("upper"))))
                ).as_matrix()
            child = joint.find("child").get("link")
            poses[child] = poses[joint.find("parent").get("link")] @ transform
            pending.remove(joint)
    for child_name in original_links:
        if not re.fullmatch(r"link_[1-5]_0", child_name):
            continue
        parent_joint = parents.get(child_name)
        if parent_joint is None:
            raise ValueError(f"Missing root joint for finger {child_name}")
        xyz = poses[child_name][:3, 3]
        roots.append(xyz)
        if int(re.fullmatch(r"link_([1-5])_0", child_name).group(1)) >= 2:
            long_roots.append(xyz)
        original_link = original_links.get(child_name)
        collision = original_link.find("collision") if original_link is not None else None
        node = collision.find("geometry/mesh") if collision is not None else None
        if node is None:
            raise ValueError(f"Missing source collision geometry for {child_name}")
        source_mesh = trimesh.load(source.parent / node.get("filename"), force="mesh")
        scale = np.fromstring(node.get("scale", "1 1 1"), sep=" ")
        radii.append(max(0.0015, min(source_mesh.extents[:2] * scale[:2]) / 2.))
    if len(roots) < 2:
        raise ValueError("Rigid palm chassis requires at least two surviving finger roots")
    roots = np.asarray(roots)
    long_roots = np.asarray(long_roots)
    if len(long_roots) == 0:
        raise ValueError("Rigid palm chassis requires at least one long-finger root")
    radius = float(np.median(radii))
    back_margin, side_margin = max(0.006, radius), max(0.004, radius)
    low, high = roots.min(axis=0), roots.max(axis=0)
    low[0] = min(-0.025, long_roots[:, 0].min() - 0.04)
    high[0] = long_roots[:, 0].min() - 0.8 * radius
    low[2] = min(0.0, low[2] - side_margin)
    high[2] = max(0.058, high[2] + side_margin)
    thickness = max(0.006, 1.2 * radius)
    center = np.array([(low[0] + high[0]) / 2, 0., (low[2] + high[2]) / 2])
    extents = np.array([high[0] - low[0], thickness, high[2] - low[2]])
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh, {"root_link": parent_link, "root_joint_count": len(roots), "root_joint_positions_m": roots.tolist(),
                  "center_m": center.tolist(), "dimensions_m": extents.tolist(),
                  "construction": "morphology_adaptive_rigid_mcp_chassis"}


def compile_hand(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    original_root = ET.parse(source).getroot()
    root.set("name", "mechanical_v2_" + root.get("name", "hand"))
    mesh_dir = destination.parent.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    outgoing = {}
    incoming = {}
    for joint in root.findall("joint"):
        outgoing.setdefault(joint.find("parent").get("link"), []).append(joint)
        incoming[joint.find("child").get("link")] = joint
    links_by_name = {link.get("name"): link for link in root.findall("link")}
    child_links = {joint.find("child").get("link") for joint in root.findall("joint")}
    root_links = sorted(set(links_by_name) - child_links)
    if len(root_links) != 1:
        raise ValueError(f"Mechanical source needs one tree root, got {root_links}")
    palm_link_name = "link_palm_chassis"
    palm_link = ET.Element("link", name=palm_link_name)
    root.append(palm_link)
    palm_joint = ET.Element("joint", name="root_to_palm_chassis", type="fixed")
    ET.SubElement(palm_joint, "parent", link=root_links[0])
    ET.SubElement(palm_joint, "child", link=palm_link_name)
    ET.SubElement(palm_joint, "origin", xyz="0 0 0", rpy="0 0 0")
    root.append(palm_joint)
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child").get("link")
        if parent.get("link") != root_links[0]:
            continue
        if re.fullmatch(r"link_[1-5]_(?:thumb_spread|mcp_spread|0)", child):
            parent.set("link", palm_link_name)
    outgoing = {}
    for joint in root.findall("joint"):
        outgoing.setdefault(joint.find("parent").get("link"), []).append(joint)
    parts, joints, palm_manifest = [], [], None
    for link in root.findall("link"):
        name = link.get("name")
        collisions = link.findall("collision")
        if not collisions and name != palm_link_name:
            continue  # A mass-only coordinate frame carries no manufactured skin.
        is_finger = re.fullmatch(r"link_[1-5]_\d+", name) is not None
        is_tip = is_finger and not outgoing.get(name)
        modules = []
        if name == palm_link_name:
            chassis, palm_manifest = palm_chassis(root, source, root_links[0], original_root)
            modules.append(("rigid_palm_chassis", chassis, 1100.0, [0.12, 0.17, 0.22, 1]))
            parts.append({"link": name, "contact_pad": True, "role": "load_bearing_palm"})
        elif is_finger:
            original = collisions[0]
            node = original.find("geometry/mesh")
            source_mesh = trimesh.load(source.parent / node.get("filename"), force="mesh")
            source_mesh.apply_scale(np.fromstring(node.get("scale", "1 1 1"), sep=" "))
            source_mesh.apply_transform(origin(original.find("origin")))
            visual = link.find("visual")
            shaft = 2 * float(visual.find("origin").get("xyz").split()[2])
            # Restore the bone's nominal radial size, rather than copying the old
            # artificial root-collision scale into the manufactured structure.
            radial_scale = float(node.get("scale", "1 1 1").split()[0])
            radius = max(0.0015, min(source_mesh.extents[:2]) / (2 * radial_scale))
            hinge_radius = radius * 0.90
            next_joint = next((j for j in outgoing.get(name, []) if j.find("child").get("link").startswith(name.rsplit("_", 1)[0] + "_")), None)
            span = float(next_joint.find("origin").get("xyz").split()[2]) if next_joint is not None else shaft
            span = max(span, 2.5 * hinge_radius)
            body_len = span - 2 * hinge_radius
            body = trimesh.creation.box(extents=[1.7 * radius, 1.6 * radius, body_len])
            body.apply_translation([0, 0, span / 2])
            modules.append(("finger_pad" if is_tip else "rigid_link", body, 1100.0, [0.12, 0.52, 0.43, 1] if is_tip else [0.35, 0.43, 0.48, 1]))
            for tag, z in (("proximal_bearing", 0.0), ("distal_bearing", span)):
                bearing = trimesh.creation.cylinder(radius=hinge_radius, height=1.8 * radius, sections=24)
                bearing.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
                bearing.apply_translation([0, 0, z])
                modules.append((tag, bearing, 2700.0, [0.80, 0.43, 0.18, 1]))
            parts.append({"link": name, "bone_shaft_m": shaft, "hinge_span_m": span,
                          "section_m": [1.7 * radius, 1.6 * radius], "bearing_radius_m": hinge_radius,
                          "contact_pad": is_tip})
        else:
            original = collisions[0]
            node = original.find("geometry/mesh")
            source_mesh = trimesh.load(source.parent / node.get("filename"), force="mesh")
            source_mesh.apply_scale(np.fromstring(node.get("scale", "1 1 1"), sep=" "))
            source_mesh.apply_transform(origin(original.find("origin")))
            modules.append(("palm_chassis" if name == "link_palm" else "wrist_mount", source_mesh.convex_hull, 1100.0, [0.12, 0.17, 0.22, 1]))
            parts.append({"link": name, "section_m": source_mesh.extents.tolist(), "contact_pad": False})
        masses, centers, tensors = [], [], []
        for old in link.findall("visual") + link.findall("collision") + link.findall("inertial"):
            link.remove(old)
        for tag, mesh, density, color in modules:
            mesh.density = density
            filename = mesh_dir / f"{name}_{tag}.stl"
            mesh.export(filename)
            for kind in ("visual", "collision"):
                element = ET.SubElement(link, kind, name=tag)
                ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
                geometry = ET.SubElement(element, "geometry")
                ET.SubElement(geometry, "mesh", filename=f"../meshes/{filename.name}")
                if kind == "visual":
                    material = ET.SubElement(element, "material", name=tag)
                    ET.SubElement(material, "color", rgba=" ".join(map(str, color)))
            masses.append(mesh.mass)
            centers.append(mesh.center_mass)
            tensors.append(mesh.moment_inertia)
        mass = sum(masses)
        center = sum(m * c for m, c in zip(masses, centers)) / mass
        tensor = np.zeros((3, 3))
        for m, c, inertia in zip(masses, centers, tensors):
            offset = c - center
            tensor += inertia + m * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(inertial, "inertia", **{k: str(tensor[i, j]) for k, i, j in (("ixx",0,0),("ixy",0,1),("ixz",0,2),("iyy",1,1),("iyz",1,2),("izz",2,2))})
        parts[-1]["mass_kg"] = mass
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        limit.set("velocity", str(min(0.45, float(limit.get("velocity")))))
        # Explicit design targets for the first mechanical simulation model.
        # Replace these with measured actuator specifications at hardware selection.
        is_base = "spread" in joint.get("name") or "link_0_0_to" in joint.get("name")
        limit.set("effort", "0.5" if is_base else "0.3")
        child = joint.find("child").get("link")
        digit = re.search(r"link_(\d+)_", child)
        joints.append({"joint": joint.get("name"), "child": child,
                       "axis": joint.find("axis").get("xyz"), "zero_rad": 0.0,
                       "limits_rad": [float(limit.get("lower")), float(limit.get("upper"))],
                       "transmission": "remote_antagonistic_tendon_proxy", "digit": int(digit.group(1)) if digit else None,
                       "actuator_mount": "forearm_rack", "tendon_routing": "through_palm_to_joint_pulley",
                       "pulley_radius_m": 0.003, "actuator_sizing_status": "unselected",
                       "effort_limit_nm": float(limit.get("effort")),
                       "tendon_differential_force_at_limit_n": float(limit.get("effort")) / 0.003})
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)
    manifest = {"version": 2, "source": str(source), "urdf": str(destination), "parts": parts, "joint_mapping": joints,
                "actuated_joint_count": len(joints), "mass_kg": sum(p["mass_kg"] for p in parts),
                "rigid_palm": palm_manifest,
                "scope": "Rigid mechanical simulation proxies. Tendon routing is a packaging plan, not elastic cable dynamics. Motor sizing is pending."}
    destination.with_suffix(".mapping.json").write_text(json.dumps(manifest, indent=2))
    from mechanical_contact_model import upgrade_contact_model
    return upgrade_contact_model(source, destination, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    result = compile_hand(args.source, args.destination)
    print(json.dumps({"joints": result["actuated_joint_count"], "mass_kg": result["mass_kg"]}))
