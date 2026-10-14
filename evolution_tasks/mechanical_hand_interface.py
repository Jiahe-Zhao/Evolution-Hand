"""Topology-derived mechanical hand interface for all task controllers."""
from pathlib import Path
import xml.etree.ElementTree as ET

TIP_HINTS = ("tip", "link_2_3", "link_3_3", "link_4_3", "link_5_3")

def discover(urdf_path):
    root = ET.parse(urdf_path).getroot()
    links = {node.get("name") for node in root.findall("link")}
    joints = []
    children = set()
    for node in root.findall("joint"):
        parent = node.find("parent"); child = node.find("child")
        if parent is None or child is None:
            continue
        child_name = child.get("link"); children.add(child_name)
        joints.append({"name": node.get("name"), "type": node.get("type"),
                       "parent": parent.get("link"), "child": child_name})
    actuated = [j for j in joints if j["type"] != "fixed"]
    roots = sorted(links - children)
    tips = sorted(name for name in links if any(h in name.lower() for h in TIP_HINTS))
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {roots}")
    if not actuated:
        raise ValueError("URDF has no actuated joints")
    return {"root": roots[0], "actuated_joints": actuated, "tip_links": tips,
            "link_count": len(links), "joint_count": len(joints)}

def task_bindings(urdf_path):
    data = discover(urdf_path)
    tips = data["tip_links"]
    return {"grasp": tips, "branch": tips, "forage": tips, "strike": tips,
            "controllable_joints": [j["name"] for j in data["actuated_joints"]]}
