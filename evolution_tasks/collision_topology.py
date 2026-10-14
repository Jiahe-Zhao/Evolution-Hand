"""Collision exclusions from actual URDF connectivity, independent of link names."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def collision_asset_fingerprint(urdf_path):
    """Invalidate cooked geometry when mesh contents change at the same path."""
    path = Path(urdf_path)
    data = path.read_bytes()
    digest = hashlib.sha256(data)
    root = ET.fromstring(data)
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if not filename:
            raise ValueError("Mesh reference has no filename")
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        digest.update(filename.encode("utf-8"))
        digest.update(mesh_path.read_bytes())
    return digest.hexdigest()[:24]


def collision_graph(urdf_path):
    """Validate a single rooted tree, including nonphysical coordinate frames."""
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    if not links or None in links or len(links) != len(root.findall("link")):
        raise ValueError("Missing or duplicate link names")
    parents = {}
    adjacency = {name: set() for name in links}
    joint_names = set()
    for joint in root.findall("joint"):
        parent_node, child_node = joint.find("parent"), joint.find("child")
        if parent_node is None or child_node is None:
            raise ValueError("Joint has no parent or child")
        parent, child = parent_node.get("link"), child_node.get("link")
        name = joint.get("name")
        if not name or name in joint_names or parent not in links or child not in links or child in parents:
            raise ValueError(f"Invalid joint connectivity: {name}")
        joint_names.add(name)
        parents[child] = (parent, joint.get("type"))
        adjacency[parent].add(child)
        adjacency[child].add(parent)
    roots = set(links) - set(parents)
    if len(roots) != 1:
        raise ValueError("URDF must contain exactly one root")
    for name in links:
        seen = set()
        current = name
        while current in parents:
            if current in seen:
                raise ValueError(f"Joint cycle at {name}")
            seen.add(current)
            current = parents[current][0]
        if current not in roots:
            raise ValueError(f"Disconnected link {name}")
    return links, parents, adjacency


def collision_exclusions(urdf_path, runtime_body_names):
    links, parents, adjacency = collision_graph(urdf_path)

    runtime = set(runtime_body_names)
    if runtime - links.keys():
        raise ValueError(f"Unmapped runtime bodies: {sorted(runtime - links.keys())}")
    bodies = {}
    for name in links:
        body, seen = name, set()
        while body not in runtime:
            if body in seen or body not in parents or parents[body][1] != "fixed":
                raise ValueError(f"Cannot map collision link {name} to runtime body")
            seen.add(body)
            body = parents[body][0]
        bodies[name] = body

    pairs = set()
    # Walk through nonphysical coordinate frames, but stop at the first physical
    # collision link. This filters a joint interface even when a generated
    # spread frame sits between the palm and a phalanx, while preserving all
    # palm-to-other-finger and cross-finger contacts.
    collision_links = {name for name, link in links.items() if link.find("collision") is not None}
    for source in collision_links:
        visited = {source}
        pending = list(adjacency[source])
        while pending:
            target = pending.pop()
            if target in visited:
                continue
            visited.add(target)
            if target in collision_links:
                a, b = bodies[source], bodies[target]
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
                continue
            pending.extend(adjacency[target] - visited)
    # Filter expected overlap at direct mechanical interfaces; retain all non-adjacent checks.
    for child, (parent, _joint_type) in parents.items():
        if child in collision_links and parent in collision_links:
            a, b = bodies[child], bodies[parent]
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)
