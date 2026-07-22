import os
from code_to_urdf import generate_urdf_from_dict
from class_agent import UrdfAgent
from variation import *
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg, cube_task_generate_env_cfg

HOME_DIR = os.path.expanduser("~")
ISAACLAB_ROOT = os.path.join(HOME_DIR, "IsaacLab")
EVOLUTION_TASK_ROOT = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "evolution_tasks"
)
CURRENT_AGENT_ROOT = os.path.join(EVOLUTION_TASK_ROOT, "current_evolution_agent")
CURRENT_AGENT_MESH_DIR = os.path.join(CURRENT_AGENT_ROOT, "mesh")
CURRENT_AGENT_URDF_PATH = os.path.join(CURRENT_AGENT_ROOT, "urdf", "current_agent.urdf")
CURRENT_AGENT_CFG_PATH = os.path.join(CURRENT_AGENT_ROOT, "current_hand_cfg.py")
CUBE_ENV_CFG_PATH = os.path.join(EVOLUTION_TASK_ROOT, "task_cube", "cube_env_cfg.py")

"""定义标准手部节点"""
base_geometry = {
    "geometry_type": "capsule",  # where is the center
    "geometry_radius": 0.01,
    "geometry_length": 0.035,
}

# 定义多个链接和关节的参数
links = [
    {
        "name_code": "link_1_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,  # too wide
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0],
        "joint_origin_rpy":[0, 1.57+0.3, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_1_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.022,
        "joint_parent": "link_1_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_1_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_1_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.045,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.01],
        "joint_origin_rpy":[0, 1.57+0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_2_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.028,
        "joint_parent": "link_2_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.045+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.02,
        "joint_parent": "link_2_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.028+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_2_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.048,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.02],
        "joint_origin_rpy":[0, 1.57, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_3_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_3_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.048+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.022,
        "joint_parent": "link_3_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_3_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.022+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.03],
        "joint_origin_rpy":[0, 1.57-0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_4_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.029,
        "joint_parent": "link_4_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.0035,
        "geometry_length": 0.021,
        "joint_parent": "link_4_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.029+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0025,
        "geometry_length": 0.015,
        "joint_parent": "link_4_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.021+0.0035], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.039,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.04],
        "joint_origin_rpy":[0, 1.57-0.2, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 1, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_5_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.02,
        "joint_parent": "link_5_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.039+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.003,
        "geometry_length": 0.015,
        "joint_parent": "link_5_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.02+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.002,
        "geometry_length": 0.012,
        "joint_parent": "link_5_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.015+0.003], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
]

# 创建 Agent 实例
initial_agent_hand = UrdfAgent(
    agent_code="initial_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
).to_dict()

generate_urdf_from_dict(
    initial_agent_hand,
    output_dir=CURRENT_AGENT_MESH_DIR,
    output_urdf=CURRENT_AGENT_URDF_PATH,
)


# 根据urdf生成isaaclab的对象定义文件
parse_urdf_and_generate_articulation_cfg(
    CURRENT_AGENT_URDF_PATH,
    CURRENT_AGENT_URDF_PATH,
    CURRENT_AGENT_CFG_PATH,
)

# generate tasks cfg according to current hand config
cube_task_generate_env_cfg(CURRENT_AGENT_URDF_PATH, 341, CUBE_ENV_CFG_PATH)


# 341 observation for normal human hand, need a founcation to calculate
