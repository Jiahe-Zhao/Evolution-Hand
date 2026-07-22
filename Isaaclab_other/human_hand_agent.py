from class_agent import UrdfAgent
from code_to_urdf import generate_urdf_from_dict
from mirror_agent import create_mirror_hand

from isaaclab_tool import parse_urdf_and_generate_articulation_cfg

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


# 定义多个链接和关节的参数
mirror_links = [
    {
        "name_code": "link_1_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,  # too wide
        "geometry_length": 0.04,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0],
        "joint_origin_rpy":[0, 1.57-0.3, 0],  # 1.57+x
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
        "joint_origin_rpy":[0, 1.57-0.1, 0],  # 1.57+x
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
        "joint_origin_rpy":[0, 1.57+0.1, 0],  # 1.57+x
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
        "joint_origin_rpy":[0, 1.57+0.2, 0],  # 1.57+x
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

mirror_agent_hand = UrdfAgent(
    agent_code="mirror_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
).to_dict()

# 加载模型

link_new=[
                    {
                        "name_code": "link_1_0",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.005462761537183412,
                        "geometry_length": 0.04175660019559794,
                        "joint_name": "link_0_0_to_link_1_0",
                        "joint_parent": "link_0_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1,
                            "effort": 15.0,
                            "velocity": 2.0
                        },
                        "joint_origin_translation": [
                            0.015,
                            0,
                            0
                        ],
                        "joint_origin_rpy": [
                            0,
                            1.87,
                            0
                        ]
                    },
                    {
                        "name_code": "link_1_1",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.003317952866806922,
                        "geometry_length": 0.024921855095042054,
                        "joint_name": "link_1_0_to_link_1_1",
                        "joint_parent": "link_1_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 10.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.045
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_1_2",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0020359239016335936,
                        "geometry_length": 0.01644832083484519,
                        "joint_name": "link_1_1_to_link_1_2",
                        "joint_parent": "link_1_1",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.026
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_2_0",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.005301678638367032,
                        "geometry_length": 0.05244839451756158,
                        "joint_name": "link_0_0_to_link_2_0",
                        "joint_parent": "link_0_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1,
                            "effort": 15.0,
                            "velocity": 2.0
                        },
                        "joint_origin_translation": [
                            0.015,
                            0,
                            0.01
                        ],
                        "joint_origin_rpy": [
                            0,
                            1.6700000000000002,
                            0
                        ]
                    },
                    {
                        "name_code": "link_2_1",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.004360888926281135,
                        "geometry_length": 0.02411259221028101,
                        "joint_name": "link_2_0_to_link_2_1",
                        "joint_parent": "link_2_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 10.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.049999999999999996
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_2_2",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.004165797458825918,
                        "geometry_length": 0.0179465023391822,
                        "joint_name": "link_2_1_to_link_2_2",
                        "joint_parent": "link_2_1",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.032
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_2_3",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.001599331580278339,
                        "geometry_length": 0.021446778544306804,
                        "joint_name": "link_2_2_to_link_2_3",
                        "joint_parent": "link_2_2",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.024
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_3_0",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.007297227370364142,
                        "geometry_length": 0.061814729790050815,
                        "joint_name": "link_0_0_to_link_3_0",
                        "joint_parent": "link_0_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1,
                            "effort": 15.0,
                            "velocity": 2.0
                        },
                        "joint_origin_translation": [
                            0.015,
                            0,
                            0.02
                        ],
                        "joint_origin_rpy": [
                            0,
                            1.57,
                            0
                        ]
                    },
                    {
                        "name_code": "link_3_1",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0029256350175303593,
                        "geometry_length": 0.03975132623548448,
                        "joint_name": "link_3_0_to_link_3_1",
                        "joint_parent": "link_3_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 10.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.052000000000000005
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_3_2",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0036180844463155155,
                        "geometry_length": 0.022625530624691893,
                        "joint_name": "link_3_1_to_link_3_2",
                        "joint_parent": "link_3_1",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.033
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_3_3",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0034700610051988582,
                        "geometry_length": 0.013523737942892683,
                        "joint_name": "link_3_2_to_link_3_3",
                        "joint_parent": "link_3_2",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.0255
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_4_0",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0034725906897112883,
                        "geometry_length": 0.05001971599132104,
                        "joint_name": "link_0_0_to_link_4_0",
                        "joint_parent": "link_0_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1,
                            "effort": 15.0,
                            "velocity": 2.0
                        },
                        "joint_origin_translation": [
                            0.015,
                            0,
                            0.03
                        ],
                        "joint_origin_rpy": [
                            0,
                            1.47,
                            0
                        ]
                    },
                    {
                        "name_code": "link_4_1",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.004063283297108501,
                        "geometry_length": 0.0223771837278474,
                        "joint_name": "link_4_0_to_link_4_1",
                        "joint_parent": "link_4_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 10.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.043000000000000003
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_4_2",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.00336170102499999,
                        "geometry_length": 0.01959239440871438,
                        "joint_name": "link_4_1_to_link_4_2",
                        "joint_parent": "link_4_1",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.033
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_4_3",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0029754954794193613,
                        "geometry_length": 0.018209737865675698,
                        "joint_name": "link_4_2_to_link_4_3",
                        "joint_parent": "link_4_2",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.0245
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_5_0",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.004296603820985498,
                        "geometry_length": 0.051161358432340646,
                        "joint_name": "link_0_0_to_link_5_0",
                        "joint_parent": "link_0_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1,
                            "effort": 15.0,
                            "velocity": 2.0
                        },
                        "joint_origin_translation": [
                            0.015,
                            0,
                            0.04
                        ],
                        "joint_origin_rpy": [
                            0,
                            1.37,
                            0
                        ]
                    },
                    {
                        "name_code": "link_5_1",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0031508930846492825,
                        "geometry_length": 0.013921404948606402,
                        "joint_name": "link_5_0_to_link_5_1",
                        "joint_parent": "link_5_0",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 10.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.042
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_5_2",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.0032430206139549924,
                        "geometry_length": 0.00956307498351381,
                        "joint_name": "link_5_1_to_link_5_2",
                        "joint_parent": "link_5_1",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.023
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    },
                    {
                        "name_code": "link_5_3",
                        "geometry_type": "capsule",
                        "geometry_radius": 0.002408693482127566,
                        "geometry_length": 0.011930735576268428,
                        "joint_name": "link_5_2_to_link_5_3",
                        "joint_parent": "link_5_2",
                        "joint_type": "revolute",
                        "joint_axis": [
                            1,
                            0,
                            0
                        ],
                        "joint_limit": {
                            "lower": 0,
                            "upper": 1.57,
                            "effort": 5.0,
                            "velocity": 1.5
                        },
                        "joint_origin_translation": [
                            0,
                            0,
                            0.018
                        ],
                        "joint_origin_rpy": [
                            0,
                            0,
                            0
                        ]
                    }
                ]

new_hand=UrdfAgent(
    agent_code="initial_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=link_new
).to_dict()


if __name__ == "__main__":
    import pybullet as p

    # 查看生成的 agent 数据
    # print(f"initial_agent_hand:{initial_agent_hand}")
    mirror_hand=create_mirror_hand(initial_agent_hand,"mirror_hand")
    # print(f"mirror_hand:{mirror_hand}")
    # 调用函数
    generate_urdf_from_dict(initial_agent_hand, output_dir="output_meshes", output_urdf="human_hand_402.urdf")
    generate_urdf_from_dict(mirror_hand, output_dir="output_meshes_402", output_urdf="human_mirror_hand_402.urdf")
    generate_urdf_from_dict(new_hand, output_dir="output_meshes_402", output_urdf="human_new_hand_402.urdf")
    parse_urdf_and_generate_articulation_cfg("/home/qyx/Isaaclab_other/human_hand_402.urdf","/home/qyx/Isaaclab_other/human_hand_402.urdf","/home/qyx/Isaaclab_other/human_hand_402.py")
    parse_urdf_and_generate_articulation_cfg("/home/qyx/Isaaclab_other/human_mirror_hand_402.urdf","/home/qyx/Isaaclab_other/human_mirror_hand_402.urdf","/home/qyx/Isaaclab_other/human_mirror_hand_402.py")
    physicsClient = p.connect(p.GUI)
    right = p.loadURDF("/home/qyx/Isaaclab_other/human_hand_402.urdf", [-0.25, 0, 1])
    left = p.loadURDF("/home/qyx/Isaaclab_other/human_new_hand_402.urdf", [0.25, 0, 1])

    # 设置视角
    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 1.0]
    )
    print("right:", right, "visuals:", len(p.getVisualShapeData(right) or []))
    print("left:", left, "visuals:", len(p.getVisualShapeData(left) or []))

    # 保持窗口
    while True:
        p.stepSimulation()
