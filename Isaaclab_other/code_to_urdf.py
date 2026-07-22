import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import trimesh


def generate_urdf_from_dict(agent_dict, output_dir="generated_meshes", output_urdf="robot.urdf"):
    from scipy.spatial.transform import Rotation as R

    def create_capsule_mesh(radius, length, count=(16, 16)):
        return trimesh.creation.capsule(radius=radius, height=length, count=count)

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
        return filename, origin_z

    def add_origin(parent, xyz, rpy):
        origin = ET.SubElement(parent, "origin")
        origin.set("xyz", " ".join(str(v) for v in xyz))
        origin.set("rpy", " ".join(str(v) for v in rpy))

    def add_mesh_geometry(parent, mesh_filename):
        geometry = ET.SubElement(parent, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        mesh.set("filename", mesh_filename)

    def add_inertial(parent, mass=0.1):
        inertial = ET.SubElement(parent, "inertial")
        add_origin(inertial, [0, 0, 0], [0, 0, 0])
        mass_tag = ET.SubElement(inertial, "mass")
        mass_tag.set("value", str(mass))
        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", "0.01")
        inertia.set("ixy", "0.0")
        inertia.set("ixz", "0.0")
        inertia.set("iyy", "0.01")
        inertia.set("iyz", "0.0")
        inertia.set("izz", "0.01")

    def add_visual_or_collision(link_tag, tag_name, mesh_filename, origin_z):
        tag = ET.SubElement(link_tag, tag_name)
        add_origin(tag, [0, 0, origin_z], [0, 0, 0])
        add_mesh_geometry(tag, mesh_filename)

    def create_transform_matrix(translation, rotation):
        tf = np.eye(4)
        tf[:3, 3] = translation
        tf[:3, :3] = R.from_euler("xyz", rotation).as_matrix()
        return tf

    os.makedirs(output_dir, exist_ok=True)

    robot = ET.Element("robot")
    robot.set("name", agent_dict["agent_code"])

    all_links = agent_dict["base_link"] + agent_dict["links"]
    for link_data in all_links:
        mesh_filename, origin_z = write_mesh(link_data, output_dir)
        link_tag = ET.SubElement(robot, "link")
        link_tag.set("name", link_data["name_code"])
        add_inertial(link_tag)
        add_visual_or_collision(link_tag, "visual", mesh_filename, origin_z)
        add_visual_or_collision(link_tag, "collision", mesh_filename, origin_z)

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
