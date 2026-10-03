import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import trimesh


# The generated hand keeps a fixed action interface, so the thumb opposition
# hinge is deliberately bounded to a usable single-axis approximation.  A
# true human thumb uses several coupled CMC axes; allowing a single hinge to
# sweep to 150 degrees makes it pass behind the palm while remaining legal in
# the URDF.
THUMB_OPPOSITION_LOWER_RAD = -2.0943951023931953  # -120 degrees
THUMB_OPPOSITION_UPPER_RAD = 0.0
FLEXION_AXIS = [1, 0, 0]
# Keep the fixed palm physically active, but hide its box-like helper mesh by default.
# Set EVOLUTION_HIDE_PALM_VISUAL=0 to show it for collision/debug inspection.
HIDE_FIXED_PALM_VISUAL = os.environ.get("EVOLUTION_HIDE_PALM_VISUAL", "1") != "0"


def closed_capsule_mesh(radius, length, sagitta_ratio=0.0):
    """Build explicit poles and rings so STL round-trips keep closed endcaps."""
    if not all(math.isfinite(v) and v > 0 for v in (radius, length)):
        raise ValueError("Capsule radius and shaft length must be finite and positive.")
    if not math.isfinite(sagitta_ratio) or not 0.0 <= sagitta_ratio <= 0.15:
        raise ValueError("Phalanx sagitta ratio must lie in [0, 0.15].")
    radial_segments, cap_segments, shaft_segments = 32, 16, 32
    rings = []
    for angle in np.linspace(-math.pi / 2, 0.0, cap_segments + 1)[1:]:
        rings.append((radius * math.cos(angle), -length / 2 + radius * math.sin(angle)))
    rings.extend((radius, z) for z in np.linspace(-length / 2, length / 2, shaft_segments + 1)[1:])
    for angle in np.linspace(0.0, math.pi / 2, cap_segments + 1)[1:-1]:
        rings.append((radius * math.cos(angle), length / 2 + radius * math.sin(angle)))
    vertices = [[0.0, 0.0, -length / 2 - radius]]
    for ring_radius, z in rings:
        for theta in np.arange(radial_segments) * (2 * math.pi / radial_segments):
            vertices.append([ring_radius * math.cos(theta), ring_radius * math.sin(theta), z])
    top = len(vertices)
    vertices.append([0.0, 0.0, length / 2 + radius])
    faces = []
    for j in range(radial_segments):
        following = (j + 1) % radial_segments
        faces.append([0, 1 + following, 1 + j])
        for ring in range(len(rings) - 1):
            a = 1 + ring * radial_segments + j
            b = 1 + ring * radial_segments + following
            faces.extend([[a, b, a + radial_segments], [b, b + radial_segments, a + radial_segments]])
        last = 1 + (len(rings) - 1) * radial_segments
        faces.append([last + j, last + following, top])
    vertices = np.asarray(vertices)
    t = np.clip(vertices[:, 2] / length + 0.5, 0.0, 1.0)
    vertices[:, 1] -= 4 * sagitta_ratio * length * t * (1 - t)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
        raise ValueError("Generated capsule is not a closed oriented solid.")
    return mesh


def curved_capsule_mesh(radius, length, sagitta_ratio):
    """A palmar bow with fixed joint endpoints; ratio is a simulation prior."""
    if not math.isfinite(sagitta_ratio) or not 0.0 <= sagitta_ratio <= 0.15:
        raise ValueError("Phalanx sagitta ratio must lie in [0, 0.15].")
    return closed_capsule_mesh(radius, length, sagitta_ratio)


def generate_urdf_from_dict(agent_dict, output_dir="generated_meshes", output_urdf="robot.urdf"):
    from scipy.spatial.transform import Rotation as R

    def create_capsule_mesh(radius, length, count=(16, 16)):
        return closed_capsule_mesh(radius, length)

    def create_palmar_surface_mesh(mirror_axis=None, lattice=False):
        """Build one smooth, tapered palm shell in the hand frame."""
        # x follows the metacarpals from wrist to knuckle line.  Each station is
        # (x, z center, half-width in y, half-thickness in z), yielding a broad
        # Australopithecus-like palm rather than independent spherical pads.
        stations = [
            (-0.025, -0.008, 0.018, 0.010),
            (-0.018, -0.003, 0.025, 0.013),
            (-0.010, 0.003, 0.032, 0.015),
            (-0.003, 0.008, 0.036, 0.017),
            (0.004, 0.012, 0.038, 0.018),
            (0.011, 0.017, 0.039, 0.018),
            (0.018, 0.022, 0.038, 0.017),
            (0.026, 0.027, 0.036, 0.015),
            (0.033, 0.031, 0.033, 0.012),
            (0.040, 0.035, 0.028, 0.009),
        ]
        ring_count = 48
        vertices = []
        for x, z_center, half_width, half_thickness in stations:
            for ring_index in range(ring_count):
                angle = 2.0 * math.pi * ring_index / ring_count
                vertices.append(
                    [x, half_width * math.cos(angle), z_center + half_thickness * math.sin(angle)]
                )

        faces = []
        for station_index in range(len(stations) - 1):
            start = station_index * ring_count
            next_start = (station_index + 1) * ring_count
            for ring_index in range(ring_count):
                # A sparse set of longitudinal and transverse bands makes a
                # porous visual membrane while the collision remains complete.
                if lattice and station_index not in (0, 3, 6, 8) and ring_index % 8 not in (0, 1):
                    continue
                next_index = (ring_index + 1) % ring_count
                faces.append([start + ring_index, start + next_index, next_start + next_index])
                faces.append([start + ring_index, next_start + next_index, next_start + ring_index])

        # Close wrist and distal ends so both the visual and collision meshes are watertight.
        wrist_center = len(vertices)
        vertices.append([stations[0][0], 0.0, stations[0][1]])
        distal_center = len(vertices)
        vertices.append([stations[-1][0], 0.0, stations[-1][1]])
        for ring_index in range(ring_count):
            if lattice and ring_index % 8 not in (0, 1):
                continue
            next_index = (ring_index + 1) % ring_count
            faces.append([wrist_center, next_index, ring_index])
            last_start = (len(stations) - 1) * ring_count
            faces.append([distal_center, last_start + ring_index, last_start + next_index])

        palm = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
        palm = palm.smoothed()
        if mirror_axis == "x":
            palm.apply_scale([-1.0, 1.0, 1.0])
        return palm

    def write_mesh(link_data, output_dir):
        name_code = link_data["name_code"]
        geometry_type = link_data.get("geometry_type")
        geometry_radius = float(link_data.get("geometry_radius", 0.1))
        geometry_length = float(link_data.get("geometry_length", 0.1))
        geometry_size = link_data.get("geometry_size")

        if geometry_type == "capsule":
            curvature = float(link_data.get("phalanx_sagitta_ratio", 0.0))
            mesh = (curved_capsule_mesh(geometry_radius, geometry_length, curvature)
                    if curvature else create_capsule_mesh(geometry_radius, geometry_length))
            filename = os.path.join(output_dir, f"{name_code}_capsule.stl")
            origin_z = geometry_length / 2.0
        elif geometry_type in {"palmar_surface", "palmar_membrane"}:
            full_palm = create_palmar_surface_mesh(link_data.get("geometry_mirror_axis"))
            mesh = create_palmar_surface_mesh(link_data.get("geometry_mirror_axis"), lattice=True) if geometry_type == "palmar_membrane" else full_palm
            mesh_suffix = "palmar_membrane" if geometry_type == "palmar_membrane" else "palmar_surface"
            filename = os.path.join(output_dir, f"{name_code}_{mesh_suffix}.stl")
            collision_filename = os.path.join(output_dir, f"{name_code}_palmar_collision.stl")
            # The full convex hull defines the hand boundary even though the
            # visible membrane intentionally has transparent gaps.
            full_palm.convex_hull.export(collision_filename)
            # The palm mesh is already authored in the link frame.
            origin_z = 0.0
        elif geometry_type == "cylinder":
            mesh = trimesh.creation.cylinder(radius=geometry_radius, height=geometry_length, sections=32)
            filename = os.path.join(output_dir, f"{name_code}_cylinder.stl")
            origin_z = geometry_length / 2.0
        elif geometry_type == "box":
            if geometry_size is not None:
                geometry_size = [float(v) for v in geometry_size]
                if len(geometry_size) != 3:
                    raise ValueError(f"Box geometry_size must have 3 elements for {name_code}")
            else:
                geometry_size = [geometry_radius] * 3
            size = np.asarray(geometry_size, dtype=float)
            mesh = trimesh.creation.box(extents=size)
            filename = os.path.join(output_dir, f"{name_code}_box.stl")
            origin_z = geometry_size[2] / 2.0
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

        mesh.export(filename)
        if geometry_type not in {"palmar_surface", "palmar_membrane"}:
            collision_filename = filename
        return filename, collision_filename, origin_z

    def add_origin(parent, xyz, rpy):
        origin = ET.SubElement(parent, "origin")
        origin.set("xyz", " ".join(str(v) for v in xyz))
        origin.set("rpy", " ".join(str(v) for v in rpy))

    def add_mesh_geometry(parent, mesh_filename):
        geometry = ET.SubElement(parent, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", mesh_filename)

    def compute_inertia(link_data):
        geometry_type = link_data.get("geometry_type")
        density = float(link_data.get("density", 850.0))
        radius = float(link_data.get("geometry_radius", 0.005))
        length = float(link_data.get("geometry_length", 0.02))
        size = link_data.get("geometry_size")

        if geometry_type == "box":
            if size is None:
                size = [radius, radius, radius]
            sx, sy, sz = [float(v) for v in size]
            mass = density * sx * sy * sz
            ixx = mass * (sy**2 + sz**2) / 12.0
            iyy = mass * (sx**2 + sz**2) / 12.0
            izz = mass * (sx**2 + sy**2) / 12.0
        elif geometry_type == "cylinder":
            mass = density * math.pi * radius * radius * length
            ixx = mass * (3 * radius**2 + length**2) / 12.0
            iyy = ixx
            izz = 0.5 * mass * radius**2
        elif geometry_type == "capsule":
            cyl_mass = density * math.pi * radius * radius * length
            sph_mass = density * (4.0 / 3.0) * math.pi * radius**3
            mass = cyl_mass + sph_mass
            ixx_cyl = cyl_mass * (3 * radius**2 + length**2) / 12.0
            izz_cyl = 0.5 * cyl_mass * radius**2
            # sph_mass is the TOTAL mass of both hemispheres. Their first
            # moments add the 3*length*radius/8 translation cross term.
            i_sphere = 0.4 * sph_mass * radius**2
            ixx = ixx_cyl + sph_mass * (
                0.4 * radius**2 + length**2 / 4.0 + 3.0 * length * radius / 8.0
            )
            iyy = ixx
            izz = izz_cyl + i_sphere
        else:
            mass = float(link_data.get("mass", 0.08))
            ixx = iyy = izz = 0.0001

        if geometry_type in {"box", "cylinder", "capsule"} and "mass" in link_data:
            specified_mass = float(link_data["mass"])
            if not math.isfinite(specified_mass) or specified_mass <= 0:
                raise ValueError("Link mass must be finite and positive.")
            scale = specified_mass / mass
            mass, ixx, iyy, izz = specified_mass, ixx * scale, iyy * scale, izz * scale
        return max(mass, 1e-6), max(ixx, 1e-12), max(iyy, 1e-12), max(izz, 1e-12)

    def add_inertial(parent, link_data):
        curvature = float(link_data.get("phalanx_sagitta_ratio", 0.0))
        if link_data.get("geometry_type") == "capsule" and curvature:
            length = float(link_data["geometry_length"])
            mesh = curved_capsule_mesh(float(link_data["geometry_radius"]), length, curvature)
            mesh.density = float(link_data.get("density", 850.0))
            if "mass" in link_data:
                mesh.density = float(link_data["mass"]) / mesh.volume
            inertial = ET.SubElement(parent, "inertial")
            add_origin(inertial, mesh.center_mass + [0.0, 0.0, length / 2.0], [0, 0, 0])
            ET.SubElement(inertial, "mass", value=str(mesh.mass))
            tensor = mesh.moment_inertia
            ET.SubElement(inertial, "inertia", **{
                key: str(tensor[i, j]) for key, i, j in
                (("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                 ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2))
            })
            return
        mass, ixx, iyy, izz = compute_inertia(link_data)
        inertial = ET.SubElement(parent, "inertial")
        geometry_type = link_data.get("geometry_type")
        if geometry_type in {"capsule", "cylinder"}:
            center_z = float(link_data.get("geometry_length", 0.02)) / 2.0
        elif geometry_type == "box":
            size = link_data.get("geometry_size", [link_data.get("geometry_radius", 0.005)] * 3)
            center_z = float(size[2]) / 2.0
        else:
            center_z = 0.0
        add_origin(inertial, [0, 0, center_z], [0, 0, 0])
        mass_tag = ET.SubElement(inertial, "mass")
        mass_tag.set("value", str(mass))
        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", str(ixx))
        inertia.set("ixy", "0.0")
        inertia.set("ixz", "0.0")
        inertia.set("iyy", str(iyy))
        inertia.set("iyz", "0.0")
        inertia.set("izz", str(izz))

    def add_visual_or_collision(link_tag, tag_name, mesh_filename, origin_z, visual_color=None, material_name=None):
        tag = ET.SubElement(link_tag, tag_name)
        add_origin(tag, [0, 0, origin_z], [0, 0, 0])
        add_mesh_geometry(tag, mesh_filename)
        if tag_name == "visual" and visual_color is not None:
            material = ET.SubElement(tag, "material")
            material.set("name", material_name or f"{link_tag.get('name')}_material")
            color = ET.SubElement(material, "color")
            color.set("rgba", " ".join(str(v) for v in visual_color))

    def create_transform_matrix(translation, rotation):
        tf = np.eye(4)
        tf[:3, 3] = translation
        tf[:3, :3] = R.from_euler("xyz", rotation).as_matrix()
        return tf

    os.makedirs(output_dir, exist_ok=True)
    output_urdf_dir = os.path.dirname(os.path.abspath(output_urdf))
    os.makedirs(output_urdf_dir, exist_ok=True)

    robot = ET.Element("robot")
    robot.set("name", agent_dict["agent_code"])

    # Never silently add removed phalanges back into an evolved topology.
    # Historical reconstructions may explicitly opt into the legacy repair.
    generated_links = [dict(link) for link in agent_dict["links"]]
    thumb_links = {link["name_code"]: link for link in generated_links if link["name_code"].startswith("link_1_")}
    legacy_thumb_repair = bool(agent_dict.get("allow_legacy_thumb_repair", False))
    if "link_1_0" not in thumb_links and not legacy_thumb_repair:
        raise ValueError("Missing thumb root; refusing to invent an unrepresented digit")
    if "link_1_0" not in thumb_links:
        thumb_root = {
            "name_code": "link_1_0",
            "geometry_type": "capsule",
            "geometry_radius": 0.005,
            "geometry_length": 0.040,
            "joint_name": "link_0_0_to_link_1_0",
            "joint_parent": "link_0_0",
            "joint_type": "revolute",
            "joint_axis": [1, 0, 0],
            "joint_limit": {"lower": 0.0, "upper": 1.0, "effort": 15.0, "velocity": 2.0},
            "joint_origin_translation": [0.015, 0.0, 0.0],
            "joint_origin_rpy": [0.0, 1.87, 0.0],
        }
        generated_links.append(thumb_root)
        thumb_links["link_1_0"] = thumb_root

    # A human-like thumb needs a proximal and distal flexion chain in addition
    # to opposition. Restore missing historical segments with conservative
    # dimensions derived from their parent, while future evolution is stopped
    # from deleting them by tools.remove_link().
    for segment_index in ((1, 2) if legacy_thumb_repair else ()):
        name = f"link_1_{segment_index}"
        if name in thumb_links:
            continue
        parent_name = f"link_1_{segment_index - 1}"
        parent_link = thumb_links[parent_name]
        parent_length = float(parent_link.get("geometry_length", 0.024))
        parent_radius = float(parent_link.get("geometry_radius", 0.004))
        recovered_link = {
            "name_code": name,
            "geometry_type": parent_link.get("geometry_type", "capsule"),
            "geometry_radius": max(0.0025, parent_radius * 0.85),
            "geometry_length": max(0.012, parent_length * 0.65),
            "joint_name": f"{parent_name}_to_{name}",
            "joint_parent": parent_name,
            "joint_type": "revolute",
            "joint_axis": [1, 0, 0],
            "joint_limit": {"lower": 0.0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
            "joint_origin_translation": [0.0, 0.0, parent_length + parent_radius],
            "joint_origin_rpy": [0.0, 0.0, 0.0],
        }
        generated_links.append(recovered_link)
        thumb_links[name] = recovered_link

    # The source representation uses a short fourth long-finger capsule as a
    # fingertip pad. Keep that collision/visual geometry, but weld it to the
    # distal phalanx: a human long finger has MCP, PIP and DIP flexion joints.
    for digit in range(2, 6):
        distal_pad_name = f"link_{digit}_3"
        for link in generated_links:
            if link.get("name_code") == distal_pad_name:
                link["joint_type"] = "fixed"
                break

    all_links = agent_dict["base_link"] + generated_links
    base_link_names = {link["name_code"] for link in agent_dict["base_link"]}
    link_names = {link["name_code"] for link in generated_links}
    finger_root_names = {f"link_{digit}_0" for digit in range(1, 6)}
    palm_root_names = {
        link["name_code"]
        for link in generated_links
        if link["name_code"] in finger_root_names and link.get("joint_parent") == "link_0_0"
    }
    # Every evolved hand is rendered with the same passive palm that was
    # validated in V3.  It adds no degree of freedom, while giving the five
    # fingers a common mechanical parent instead of five independent wrists.
    add_fixed_palm = (
        "link_0_0" in base_link_names
        and "link_palm" not in link_names
        and bool(palm_root_names)
    )
    is_mirrored = bool(agent_dict.get("is_mirrored", False)) or (
        "mirror" in str(agent_dict.get("agent_code", "")).lower()
    )

    # The original thumb root was a single flexion hinge. Keep that hinge and
    # add one CMC spread hinge at the same fixed root contact point. The CMC
    # is limited to a human functional arc rather than a mechanically possible
    # 180-degree sweep; the original three thumb hinges remain the flexion
    # chain.
    thumb_spread_link = "link_1_thumb_spread"
    thumb_spread_joint = "link_1_thumb_spread_joint"
    has_thumb_root = any(link.get("name_code") == "link_1_0" for link in generated_links)
    add_thumb_spread = has_thumb_root and thumb_spread_link not in link_names
    # Human long-finger MCP joints have a flexion/extension axis and a smaller
    # abduction/adduction axis.  The existing root hinge supplies flexion; add
    # this short intermediary link for the latter.  Ranges are deliberately
    # conservative, in radians: index +/-20, middle +/-10, ring +/-15 and
    # little +/-20 degrees.  This preserves a stable grip while allowing the
    # fan to open and cup around objects.
    mcp_spread_specs = {
        2: {"lower": -0.35, "upper": 0.35, "effort": 8.0, "velocity": 1.5},
        3: {"lower": -0.17, "upper": 0.17, "effort": 8.0, "velocity": 1.5},
        4: {"lower": -0.26, "upper": 0.26, "effort": 8.0, "velocity": 1.5},
        5: {"lower": -0.35, "upper": 0.35, "effort": 8.0, "velocity": 1.5},
    }
    mcp_spread_links = {digit: f"link_{digit}_mcp_spread" for digit in mcp_spread_specs}
    mcp_spread_joints = {digit: f"link_{digit}_mcp_spread_joint" for digit in mcp_spread_specs}
    add_mcp_spread = {
        digit: f"link_{digit}_0" in palm_root_names and mcp_spread_links[digit] not in link_names
        for digit in mcp_spread_specs
    }
    root_origins = {
        # Fixed thumb-root contact point on the radial/proximal palm edge.
        # Radial CMC anchor: the straight thumb must not intersect index/middle
        # rays. Negative opposition then swings toward the finger row.
        1: ([0.0, 0.0, -0.006], [0.0, math.pi / 2.0, 0.0]),
        2: ([0.015, 0.0, 0.010], [0.0, 1.67, 0.0]),
        3: ([0.015, 0.0, 0.020], [0.0, 1.57, 0.0]),
        4: ([0.015, 0.0, 0.030], [0.0, 1.47, 0.0]),
        5: ([0.015, 0.0, 0.040], [0.0, 1.37, 0.0]),
    }

    reference_root_origins = dict(root_origins)
    reference_root_origins[1] = ([0.015, 0.0, 0.0], [0.0, 1.87, 0.0])
    source_roots = {link["name_code"]: link for link in generated_links}

    def get_root_pose(digit):
        translation, rpy = root_origins[digit]
        reference_translation, reference_rpy = reference_root_origins[digit]
        if is_mirrored:
            translation = [-translation[0], *translation[1:]]
            rpy = [rpy[0], -rpy[1], -rpy[2]]
            reference_translation = [-reference_translation[0], *reference_translation[1:]]
            reference_rpy = [reference_rpy[0], -reference_rpy[1], -reference_rpy[2]]
        source = source_roots[f"link_{digit}_0"]
        # Preserve the calibrated thumb frame while applying the actual
        # evolved translation/rotation, rather than overwriting mutations.
        translation = np.asarray(translation) + np.asarray(source["joint_origin_translation"]) - reference_translation
        rotation = R.from_euler("xyz", rpy) * R.from_euler("xyz", reference_rpy).inv() * R.from_euler("xyz", source["joint_origin_rpy"])
        return translation.tolist(), rotation.as_euler("xyz").tolist()

    # Keep the wrist volume, but extend palm support to the current MCP
    # anchors. Finger lengths/topology remain separate collision solids.
    palm_low = np.array([-0.020, -0.010, 0.0])
    palm_high = np.array([0.020, 0.010, 0.058])
    for digit in range(2, 6):
        if f"link_{digit}_0" not in palm_root_names:
            continue
        anchor = np.asarray(get_root_pose(digit)[0])
        radius = float(source_roots[f"link_{digit}_0"]["geometry_radius"])
        palm_low = np.minimum(palm_low, anchor - radius)
        palm_high = np.maximum(palm_high, anchor + radius)
    palm_size = palm_high - palm_low
    palm_joint_translation = (palm_low + palm_high) / 2 - np.array([0, 0, palm_size[2] / 2])
    palm_data = {
        "name_code": "link_palm",
        "geometry_type": "box",
        "geometry_size": palm_size.tolist(),
        "mass": 0.16,
        "visual_color": [0.72, 0.49, 0.30, 1.0],
    }
    for link_data in all_links:
        mesh_filename_abs, collision_filename_abs, origin_z = write_mesh(link_data, output_dir)
        mesh_filename = os.path.relpath(mesh_filename_abs, output_urdf_dir)
        collision_filename = os.path.relpath(collision_filename_abs, output_urdf_dir)
        link_tag = ET.SubElement(robot, "link")
        link_tag.set("name", link_data["name_code"])
        add_inertial(link_tag, link_data)
        add_visual_or_collision(link_tag, "visual", mesh_filename, origin_z, link_data.get("visual_color"))
        add_visual_or_collision(link_tag, "collision", collision_filename, origin_z)

        # Extra root-link geometries are retained by IsaacLab's URDF importer,
        # unlike visuals attached through a merged fixed child joint.  This is
        # used for the translucent palmar membrane around the moving skeleton.
        for overlay_index, overlay in enumerate(link_data.get("visual_overlays", [])):
            overlay_data = dict(overlay)
            overlay_name = overlay_data.pop("name_code", f"overlay_{overlay_index}")
            overlay_data["name_code"] = f"{link_data['name_code']}_{overlay_name}"
            overlay_mesh_abs, overlay_collision_abs, overlay_origin_z = write_mesh(overlay_data, output_dir)
            overlay_mesh = os.path.relpath(overlay_mesh_abs, output_urdf_dir)
            overlay_collision = os.path.relpath(overlay_collision_abs, output_urdf_dir)
            add_visual_or_collision(
                link_tag,
                "visual",
                overlay_mesh,
                overlay_origin_z,
                overlay_data.get("visual_color"),
                material_name=f"{link_data['name_code']}_{overlay_name}_material",
            )
            if overlay_data.get("collision_enabled", False):
                add_visual_or_collision(link_tag, "collision", overlay_collision, overlay_origin_z)

    if add_fixed_palm:
        mesh_filename_abs, collision_filename_abs, origin_z = write_mesh(palm_data, output_dir)
        palm_tag = ET.SubElement(robot, "link")
        palm_tag.set("name", "link_palm")
        add_inertial(palm_tag, palm_data)
        if not HIDE_FIXED_PALM_VISUAL:
            add_visual_or_collision(
                palm_tag,
                "visual",
                os.path.relpath(mesh_filename_abs, output_urdf_dir),
                origin_z,
                palm_data["visual_color"],
            )
        # The palm is a load-bearing part of a human grasp.  Keep its fixed
        # kinematics but give it the same physical collision geometry as the
        # visible palmar arch, so objects can be supported by the palm.
        add_visual_or_collision(
            palm_tag,
            "collision",
            os.path.relpath(collision_filename_abs, output_urdf_dir),
            origin_z,
        )

        fixed_joint = ET.SubElement(robot, "joint")
        fixed_joint.set("name", "link_0_0_to_link_palm")
        fixed_joint.set("type", "fixed")
        parent = ET.SubElement(fixed_joint, "parent")
        parent.set("link", "link_0_0")
        child = ET.SubElement(fixed_joint, "child")
        child.set("link", "link_palm")
        add_origin(fixed_joint, palm_joint_translation.tolist(), [0.0, 0.0, 0.0])

    if add_thumb_spread:
        spread_tag = ET.SubElement(robot, "link")
        spread_tag.set("name", thumb_spread_link)
        spread_data = {
            "name_code": thumb_spread_link,
            "mass": 0.0001,
            "geometry_type": "box",
            "geometry_size": [0.002, 0.002, 0.002],
        }
        add_inertial(spread_tag, spread_data)
        # An inertial coordinate frame has no skin surface. Keep its mass
        # and articulation, but let the physical palm and phalanges collide.
        spread_joint = ET.SubElement(robot, "joint")
        spread_joint.set("name", thumb_spread_joint)
        spread_joint.set("type", "revolute")
        parent = ET.SubElement(spread_joint, "parent")
        parent.set("link", "link_palm" if add_fixed_palm else "link_0_0")
        child = ET.SubElement(spread_joint, "child")
        child.set("link", thumb_spread_link)
        origin_translation, origin_rpy = get_root_pose(1)
        if add_fixed_palm:
            origin_translation = (np.asarray(origin_translation) - palm_joint_translation).tolist()
        add_origin(spread_joint, origin_translation, origin_rpy)
        axis = ET.SubElement(spread_joint, "axis")
        # The hand plane is X-Z and Y is its normal. A CMC opposition sweep
        # therefore rotates in the palm plane. Mirrored hands use the opposite
        # signed axis so both hands travel toward their own ulnar side.
        axis.set("xyz", "0 1 0" if not is_mirrored else "0 -1 0")
        limit = ET.SubElement(spread_joint, "limit")
        limit.set("lower", str(THUMB_OPPOSITION_LOWER_RAD))
        limit.set("upper", str(THUMB_OPPOSITION_UPPER_RAD))
        limit.set("effort", "15.0")
        limit.set("velocity", "2.0")

    for digit, spec in mcp_spread_specs.items():
        if not add_mcp_spread[digit]:
            continue
        spread_link = mcp_spread_links[digit]
        spread_tag = ET.SubElement(robot, "link")
        spread_tag.set("name", spread_link)
        spread_data = {
            "name_code": spread_link,
            "mass": 0.0001,
            "geometry_type": "box",
            "geometry_size": [0.002, 0.002, 0.002],
        }
        add_inertial(spread_tag, spread_data)
        # Coordinate frames retain inertia but do not add artificial skin.
        spread_joint = ET.SubElement(robot, "joint")
        spread_joint.set("name", mcp_spread_joints[digit])
        spread_joint.set("type", "revolute")
        parent = ET.SubElement(spread_joint, "parent")
        parent.set("link", "link_palm" if add_fixed_palm else "link_0_0")
        child = ET.SubElement(spread_joint, "child")
        child.set("link", spread_link)
        origin_translation, origin_rpy = get_root_pose(digit)
        if add_fixed_palm:
            origin_translation = (np.asarray(origin_translation) - palm_joint_translation).tolist()
        add_origin(spread_joint, origin_translation, origin_rpy)
        axis = ET.SubElement(spread_joint, "axis")
        # X is the proximal-distal direction at the MCP roots. Rotating around
        # X splays the fingers in the transverse palm direction.
        axis.set("xyz", "1 0 0" if not is_mirrored else "-1 0 0")
        limit = ET.SubElement(spread_joint, "limit")
        limit.set("lower", str(spec["lower"]))
        limit.set("upper", str(spec["upper"]))
        limit.set("effort", str(spec["effort"]))
        limit.set("velocity", str(spec["velocity"]))

    for link_data in generated_links:
        joint_tag = ET.SubElement(robot, "joint")
        joint_tag.set("name", link_data["joint_name"])
        joint_tag.set("type", link_data.get("joint_type", "revolute"))

        parent = ET.SubElement(joint_tag, "parent")
        joint_parent = link_data.get("joint_parent", "base_link")
        if add_thumb_spread and link_data["name_code"] == "link_1_0":
            joint_parent = thumb_spread_link
        elif link_data["name_code"] in mcp_spread_links.values():
            raise RuntimeError("MCP spread links are generated internally and must not appear in agent links.")
        elif link_data["name_code"] in palm_root_names and int(link_data["name_code"].split("_")[1]) in mcp_spread_specs and add_mcp_spread[int(link_data["name_code"].split("_")[1])]:
            digit = int(link_data["name_code"].split("_")[1])
            joint_parent = mcp_spread_links[digit]
        elif add_fixed_palm and link_data["name_code"] in palm_root_names:
            joint_parent = "link_palm"
        parent.set("link", joint_parent)

        child = ET.SubElement(joint_tag, "child")
        child.set("link", link_data["name_code"])

        origin_translation = link_data.get("joint_origin_translation", [0, 0, 0])
        origin_rpy = link_data.get("joint_origin_rpy", [0, 0, 0])
        if add_thumb_spread and link_data["name_code"] == "link_1_0":
            # The root spread joint above owns the canonical thumb root pose;
            # this original hinge only contributes thumb flexion.
            origin_translation, origin_rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        elif link_data["name_code"] in palm_root_names and int(link_data["name_code"].split("_")[1]) in mcp_spread_specs and add_mcp_spread[int(link_data["name_code"].split("_")[1])]:
            # The MCP splay joint owns the root pose; the original root hinge
            # remains the finger's flexion/extension axis at that same point.
            origin_translation, origin_rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        elif link_data["name_code"] in palm_root_names:
            # Preserve V3's coordinated fan geometry even when an evolutionary
            # mutation targets a root joint.  The mirrored hand flips x/y.
            digit = int(link_data["name_code"].split("_")[1])
            origin_translation, origin_rpy = get_root_pose(digit)
            if add_fixed_palm:
                origin_translation = (np.asarray(origin_translation) - palm_joint_translation).tolist()
        add_origin(joint_tag, origin_translation, origin_rpy)

        if link_data.get("joint_type", "revolute") != "fixed":
            joint_axis = ET.SubElement(joint_tag, "axis")
            link_name = link_data["name_code"]
            # Evolution may change geometry and topology, but not the
            # anatomical meaning of a flexion hinge.  Random Euler-origin
            # changes are therefore not allowed to rotate these axes into a
            # backward-bending coordinate frame.
            if link_name.startswith("link_") and link_name.count("_") == 2:
                parts = link_name.split("_")
                if parts[1].isdigit() and 1 <= int(parts[1]) <= 5 and parts[2].isdigit() and int(parts[2]) <= 2:
                    joint_axis_value = FLEXION_AXIS
                else:
                    joint_axis_value = link_data.get("joint_axis", FLEXION_AXIS)
            else:
                joint_axis_value = link_data.get("joint_axis", FLEXION_AXIS)
            joint_axis.set("xyz", " ".join(str(v) for v in joint_axis_value))

            joint_limit_data = dict(link_data.get(
                "joint_limit",
                {"lower": -1.57, "upper": 1.57, "effort": 10.0, "velocity": 1.0},
            ))
            # Keep the evolved geometry, but use a common human-like ROM for
            # the anatomical flexion joints.  Long fingers retain MCP, PIP,
            # DIP (the short terminal pad is fixed); the values below are
            # functional flexion maxima in radians, not mechanical extremes.
            human_flexion_limits = {
                "link_1_0": 1.05,  # thumb CMC/MCP flexion, about 60 deg
                "link_1_1": 1.05,  # thumb MCP flexion, about 60 deg
                "link_1_2": 1.55,  # thumb IP flexion, about 89 deg
                "link_2_0": 1.18,  # index MCP, about 68 deg
                "link_3_0": 1.40,  # middle MCP, about 80 deg
                "link_4_0": 1.46,  # ring MCP, about 84 deg
                "link_5_0": 1.59,  # little MCP, about 91 deg
            }
            if link_name in human_flexion_limits:
                joint_limit_data.update(lower=0.0, upper=human_flexion_limits[link_name])
            elif link_name.startswith(("link_2_1", "link_3_1", "link_4_1", "link_5_1")):
                joint_limit_data.update(lower=0.0, upper=1.75)  # PIP, about 100 deg
            elif link_name.startswith(("link_2_2", "link_3_2", "link_4_2", "link_5_2")):
                joint_limit_data.update(lower=0.0, upper=1.57)  # DIP, about 90 deg
            limit = ET.SubElement(joint_tag, "limit")
            limit.set("lower", str(joint_limit_data.get("lower", -1.57)))
            limit.set("upper", str(joint_limit_data.get("upper", 1.57)))
            limit.set("effort", str(joint_limit_data.get("effort", 10.0)))
            limit.set("velocity", str(joint_limit_data.get("velocity", 1.0)))

    rough_xml = ET.tostring(robot, encoding="utf-8")
    pretty_xml = minidom.parseString(rough_xml).toprettyxml(indent="  ")
    with open(output_urdf, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"URDF saved to {output_urdf}")
