"""Coarse-grained, passive MuJoCo bristle bundles."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()

import math
import xml.etree.ElementTree as ET


def build_bristles(parent_xml_element, n_bundles=13, segments=5,
                   length=.022, root_radius=.0035):
    """Attach tapered passive bundles and return Newton joint configuration metadata."""
    if n_bundles < 1 or segments < 1 or length <= 0 or root_radius < 0:
        raise ValueError("invalid bristle dimensions")
    segment_length = length / segments
    ei = 5e-8  # Coarse-grained design parameter, not measured material data.
    stiffness = ei / segment_length
    damping = 2e-7
    metadata = {
        "joints": [],
        "geoms": [],
        "segment_lengths": [segment_length] * (n_bundles * segments),
        "k": stiffness,
        "d": damping,
        "EI": ei,
        "design_note": "EI is a coarse-grained design parameter, not measured material data.",
    }
    inward_tilt = math.atan2(max(0.,root_radius-.0016), length) if root_radius else 0.0

    for bundle in range(n_bundles):
        if n_bundles == 13:
            rr = 0.0 if bundle == 0 else root_radius * (.5 if bundle <= 6 else 1.)
            angle = 2 * math.pi * ((bundle-1) % 6) / 6 + (0 if bundle <= 6 else math.pi/6)
        else:
            rr = root_radius
            angle = 2.0 * math.pi * bundle / n_bundles
        inward_tilt = math.atan2(rr * .55, length)
        ca, sa = math.cos(angle), math.sin(angle)
        parent = parent_xml_element
        for segment in range(segments):
            name = f"bristle_{bundle:02d}_{segment:02d}"
            body = ET.SubElement(parent, "body", name=name)
            if segment == 0:
                body.set("pos", f"{rr * ca:.9g} {rr * sa:.9g} 0")
                body.set("euler", f"{-inward_tilt * sa:.9g} {inward_tilt * ca:.9g} 0")
            else:
                body.set("pos", f"0 0 {-segment_length:.9g}")
            ET.SubElement(body, "inertial", pos=f"0 0 {-segment_length / 2:.9g}",
                          mass="5e-6", diaginertia="8.1e-12 8.1e-12 5e-13")
            for axis_name, axis in (("x", "1 0 0"), ("y", "0 1 0")):
                joint_name = f"{name}_{axis_name}"
                ET.SubElement(body, "joint", name=joint_name, type="hinge", axis=axis,
                              pos="0 0 0", limited="false", damping=f"{damping:.9g}",
                              stiffness=f"{stiffness:.9g}", springref="0")
                metadata["joints"].append({"name": joint_name, "axis": axis,
                                           "k": stiffness, "d": damping})
            geom_name = f"{name}_capsule"
            radius = .00045 - .00020 * segment / max(1, segments - 1)
            ET.SubElement(body, "geom", name=geom_name, type="capsule",
                          fromto=f"0 0 0 0 0 {-segment_length:.9g}",
                          size=f"{radius:.9g}", contype="2", conaffinity="1",
                          rgba=".73 .65 .46 1", density="0", friction="0.6 0.01 0.001")
            metadata["geoms"].append(geom_name)
            if segment == segments - 1:
                tip_name = f"{name}_tip"
                ET.SubElement(body, "geom", name=tip_name, type="sphere",
                              pos=f"0 0 {-segment_length:.9g}", size="0.00025",
                              contype="2", conaffinity="1", density="0", rgba=".09 .07 .05 1")
                metadata["geoms"].append(tip_name)
            parent = body
    return metadata
