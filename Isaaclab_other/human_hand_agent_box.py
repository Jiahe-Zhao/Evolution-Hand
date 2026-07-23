from copy import deepcopy

from class_agent import UrdfAgent
from code_to_urdf import generate_urdf_from_dict
from human_hand_agent import base_geometry as base_geometry_ref
from human_hand_agent import link_new as link_new_ref
from human_hand_agent import links as links_ref
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
from mirror_agent import create_mirror_hand

base_geometry = deepcopy(base_geometry_ref)
links = deepcopy(links_ref)
link_new = deepcopy(link_new_ref)

PROXIMAL_LINKS = ("link_1_0", "link_2_0", "link_3_0", "link_4_0", "link_5_0")

# Box dimensions (meters) for each proximal phalanx: [x, y, z].
# Here z is the finger-forward direction in local link frame, so increasing z makes the box "longer".
PROXIMAL_BOX_DIMS = {
    "link_1_0": [0.008, 0.010, 0.03],
    "link_2_0": [0.009, 0.011, 0.032],
    "link_3_0": [0.009, 0.012, 0.034],
    "link_4_0": [0.009, 0.011, 0.032],
    "link_5_0": [0.008, 0.010, 0.030],
}

# Move proximal roots slightly outward from the palm so boxes are visible and do not overlap too much.
PROXIMAL_JOINT_ORIGIN = {
    "link_1_0": [0.010, -0.008, 0.000],
    "link_2_0": [0.010, -0.004, 0.010],
    "link_3_0": [0.010, 0.000, 0.020],
    "link_4_0": [0.010, 0.004, 0.030],
    "link_5_0": [0.010, 0.008, 0.040],
}

CHILD_GAP = 0.0005


def apply_proximal_box_variant(link_defs):
    for link in link_defs:
        name = link["name_code"]
        if name in PROXIMAL_LINKS:
            link["geometry_type"] = "box"
            link["geometry_size"] = PROXIMAL_BOX_DIMS[name]
            # Keep this field for backward compatibility with places that still read geometry_radius.
            link["geometry_radius"] = PROXIMAL_BOX_DIMS[name][0]
            link["joint_origin_translation"] = PROXIMAL_JOINT_ORIGIN[name]

    # Reposition first child joints so they connect to the top face of each proximal box.
    for link in link_defs:
        parent = link.get("joint_parent")
        if parent in PROXIMAL_BOX_DIMS and link["name_code"].endswith("_1"):
            parent_box_length = PROXIMAL_BOX_DIMS[parent][2]
            link["joint_origin_translation"] = [0.0, 0.0, parent_box_length + CHILD_GAP]


apply_proximal_box_variant(links)
apply_proximal_box_variant(link_new)

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

    hand_urdf = "/home/qyx/Isaaclab_other/human_hand_box.urdf"
    mirror_urdf = "/home/qyx/Isaaclab_other/human_mirror_hand_box.urdf"
    new_urdf = "/home/qyx/Isaaclab_other/human_new_hand_box.urdf"
    hand_cfg = "/home/qyx/Isaaclab_other/human_hand_box.py"
    mirror_cfg = "/home/qyx/Isaaclab_other/human_mirror_hand_box.py"

    mirror_hand = create_mirror_hand(initial_agent_hand, "mirror_hand")

    generate_urdf_from_dict(initial_agent_hand, output_dir="output_meshes_box", output_urdf=hand_urdf)
    generate_urdf_from_dict(mirror_hand, output_dir="output_meshes_mirror_box", output_urdf=mirror_urdf)
    # generate_urdf_from_dict(new_hand, output_dir="output_meshes_new_box", output_urdf=new_urdf)
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
