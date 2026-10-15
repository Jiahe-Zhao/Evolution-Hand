#!/usr/bin/env python3
"""Add modular mechanical visual parts without changing URDF physics."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


COLORS = {
    "chassis": "0.10 0.16 0.19 1",
    "shell": "0.35 0.44 0.47 1",
    "rail": "0.72 0.78 0.76 1",
    "hinge": "0.88 0.43 0.16 1",
    "pad": "0.08 0.61 0.53 1",
}


def visual(link: ET.Element, geometry: str, dimensions: dict[str, str],
           xyz: tuple[float, float, float], rpy: str, color: str) -> None:
    part = ET.SubElement(link, "visual")
    ET.SubElement(part, "origin", xyz=" ".join(f"{value:.6g}" for value in xyz), rpy=rpy)
    shape = ET.SubElement(part, "geometry")
    ET.SubElement(shape, geometry, **dimensions)
    material = ET.SubElement(part, "material", name=f"mechanical_{color}")
    ET.SubElement(material, "color", rgba=COLORS[color])


def style_hand(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    parents = {joint.find("parent").get("link") for joint in root.findall("joint")}
    for link in root.findall("link"):
        name = link.get("name", "")
        original = link.find("visual")
        if name == "link_palm":
            collision_mesh = link.find("collision/geometry/mesh")
            if original is None and collision_mesh is not None:
                part = ET.SubElement(link, "visual")
                part.append(ET.fromstring(ET.tostring(link.find("collision/origin"))))
                shape = ET.SubElement(part, "geometry")
                ET.SubElement(shape, "mesh", filename=collision_mesh.get("filename"))
                original = part
        if original is not None:
            color = "chassis" if name in ("link_palm", "link_0_0") else "shell"
            if re.fullmatch(r"link_[1-5]_\d+", name) and name not in parents:
                color = "pad"
            if "spread" in name:
                color = "hinge"
            material = ET.SubElement(original, "material", name=f"mechanical_{color}")
            ET.SubElement(material, "color", rgba=COLORS[color])

        match = re.fullmatch(r"link_([1-5])_(\d+)", name)
        if not match or original is None:
            continue
        origin = original.find("origin")
        if origin is None:
            continue
        z_center = float(origin.get("xyz", "0 0 0").split()[2])
        length = max(0.008, 2 * z_center)
        radius = min(0.005, max(0.002, length * 0.11))
        if name not in parents:
            visual(link, "box", {"size": f"{radius * 1.6:.6g} {radius * 1.2:.6g} {length * 0.48:.6g}"},
                   (0, -radius * 0.8, length * 0.68), "0 0 0", "pad")
            continue
        # Twin dorsal rails and a transverse hinge pin make the modules legible.
        for side in (-1, 1):
            visual(link, "box", {"size": f"{radius * 0.45:.6g} {radius * 0.55:.6g} {length * 0.68:.6g}"},
                   (side * radius * 0.92, 0, length * 0.52), "0 0 0", "rail")
        visual(link, "cylinder", {"radius": f"{radius * 0.82:.6g}", "length": f"{radius * 2.4:.6g}"},
               (0, 0, length * 0.08), "0 1.57079632679 0", "hinge")

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    style_hand(args.source, args.destination)
