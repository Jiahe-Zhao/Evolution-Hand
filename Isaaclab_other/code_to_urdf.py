import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import trimesh


def generate_urdf_from_dict(agent_dict, output_dir="generated_meshes", output_urdf="robot.urdf"):
    from scipy.spatial.transform import Rotation as R

    def create_capsule_mesh(radius, length, count=(16, 16)):
        return trimesh.creation.capsule(radius=radius, height=length, count=count)

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
            mesh = create_capsule_mesh(geometry_radius, geometry_length)
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
            i_sphere = 0.4 * sph_mass * radius**2
            shift = (length / 2.0) ** 2
            ixx = ixx_cyl + 2 * (i_sphere + sph_mass * shift)
            iyy = ixx
            izz = izz_cyl + 2 * i_sphere
        else:
            mass = float(link_data.get("mass", 0.08))
            ixx = iyy = izz = 0.0001

        return max(mass, 1e-4), max(ixx, 1e-6), max(iyy, 1e-6), max(izz, 1e-6)

    def add_inertial(parent, link_data):
        mass, ixx, iyy, izz = compute_inertia(link_data)
        inertial = ET.SubElement(parent, "inertial")
        add_origin(inertial, [0, 0, 0], [0, 0, 0])
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

    robot = ET.Element("robot")
    robot.set("name", agent_dict["agent_code"])

    all_links = agent_dict["base_link"] + agent_dict["links"]
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

    for link_data in agent_dict["links"]:
        joint_tag = ET.SubElement(robot, "joint")
        joint_tag.set("name", link_data["joint_name"])
        joint_tag.set("type", link_data.get("joint_type", "revolute"))

        parent = ET.SubElement(joint_tag, "parent")
        parent.set("link", link_data.get("joint_parent", "base_link"))

        child = ET.SubElement(joint_tag, "child")
        child.set("link", link_data["name_code"])

        origin_translation = link_data.get("joint_origin_translation", [0, 0, 0])
        origin_rpy = link_data.get("joint_origin_rpy", [0, 0, 0])
        add_origin(joint_tag, origin_translation, origin_rpy)

        if link_data.get("joint_type", "revolute") != "fixed":
            joint_axis = ET.SubElement(joint_tag, "axis")
            joint_axis.set("xyz", " ".join(str(v) for v in link_data.get("joint_axis", [1, 0, 0])))

            joint_limit_data = link_data.get(
                "joint_limit",
                {"lower": -1.57, "upper": 1.57, "effort": 10.0, "velocity": 1.0},
            )
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
