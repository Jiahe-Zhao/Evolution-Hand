import os
from copy import deepcopy

from class_agent import UrdfAgent
from code_to_urdf import generate_urdf_from_dict
from human_hand_agent import base_geometry as base_geometry_ref
from human_hand_agent import link_new as link_new_ref
from human_hand_agent import links as links_ref
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
from mirror_agent import create_mirror_hand

BASE_DIR = os.path.dirname(__file__)


def _box_dims_from_capsule(radius, length):
    """Map capsule params to box dims [x, y, z] while keeping parent-child spacing consistent."""
    width = max(2.0 * float(radius), 0.004)
    depth = max(2.0 * float(radius), 0.004)
    # Existing joint offsets use (parent_length + parent_radius), so keep that as box z-size.
    height = max(float(length) + float(radius), 0.006)
    return [width, depth, height]


def _convert_link_to_box(link):
    radius = float(link.get("geometry_radius", 0.003))
    length = float(link.get("geometry_length", 0.02))
    dims = _box_dims_from_capsule(radius, length)
    link["geometry_type"] = "box"
    link["geometry_size"] = dims
    # Keep compatibility with code paths that still read geometry_radius / geometry_length.
    link["geometry_radius"] = dims[0]
    link["geometry_length"] = dims[2]


def _apply_all_box(link_defs):
    for link in link_defs:
        _convert_link_to_box(link)


# Base link: capsule -> box
_base_box_dims = _box_dims_from_capsule(
    base_geometry_ref["geometry_radius"], base_geometry_ref["geometry_length"]
)
base_geometry = {
    "geometry_type": "box",
    "geometry_size": _base_box_dims,
    "geometry_radius": _base_box_dims[0],
    "geometry_length": _base_box_dims[2],
}

links = deepcopy(links_ref)
link_new = deepcopy(link_new_ref)
_apply_all_box(links)
_apply_all_box(link_new)
mirror_links = deepcopy(links)

initial_agent_hand = UrdfAgent(
    agent_code="initial_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links,
).to_dict()

mirror_agent_hand = UrdfAgent(
    agent_code="mirror_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=mirror_links,
).to_dict()

new_hand = UrdfAgent(
    agent_code="initial_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=link_new,
).to_dict()


if __name__ == "__main__":
    import pybullet as p

    hand_urdf = os.path.join(BASE_DIR, "human_hand_box_all.urdf")
    mirror_urdf = os.path.join(BASE_DIR, "human_mirror_hand_box_all.urdf")
    new_urdf = os.path.join(BASE_DIR, "human_new_hand_box_all.urdf")
    hand_cfg = os.path.join(BASE_DIR, "human_hand_box_all.py")
    mirror_cfg = os.path.join(BASE_DIR, "human_mirror_hand_box_all.py")

    mirror_hand = create_mirror_hand(initial_agent_hand, "mirror_hand")

    generate_urdf_from_dict(initial_agent_hand, output_dir="output_meshes_box_all", output_urdf=hand_urdf)
    generate_urdf_from_dict(mirror_hand, output_dir="output_meshes_mirror_box_all", output_urdf=mirror_urdf)
    # generate_urdf_from_dict(new_hand, output_dir="output_meshes_new_box_all", output_urdf=new_urdf)
    parse_urdf_and_generate_articulation_cfg(hand_urdf, hand_urdf, hand_cfg)
    # parse_urdf_and_generate_articulation_cfg(mirror_urdf, mirror_urdf, mirror_cfg)

    physics_client = p.connect(p.GUI)
    right = p.loadURDF(hand_urdf, [-0.25, 0, 1])
    left = p.loadURDF(mirror_urdf, [0.25, 0, 1])

    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0],
    )
    print("right:", right, "visuals:", len(p.getVisualShapeData(right) or []))
    print("left:", left, "visuals:", len(p.getVisualShapeData(left) or []))

    while True:
        p.stepSimulation()
