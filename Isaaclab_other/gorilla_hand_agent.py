from class_agent import UrdfAgent
from code_to_urdf import generate_urdf_from_dict

"""gorilla hand define"""

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
        "joint_origin_translation":[0.01, 0, 0],
        "joint_origin_rpy":[0, 1.57+0.3, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 0, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_1_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
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
        "joint_origin_translation":[0, 0, 0.022+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.09,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.01, 0, 0.01],
        "joint_origin_rpy":[0, 1.57+0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 0, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_2_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.006,
        "geometry_length": 0.05,
        "joint_parent": "link_2_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.09+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_2_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.03,
        "joint_parent": "link_2_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.05+0.006], # parent's radius and length
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
        "joint_origin_translation":[0, 0, 0.03+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.086,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.02],
        "joint_origin_rpy":[0, 1.57, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 0, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_3_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.008,
        "geometry_length": 0.05,
        "joint_parent": "link_3_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.086+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_3_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.004,
        "geometry_length": 0.04,
        "joint_parent": "link_3_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.05+0.008], # parent's radius and length
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
        "joint_origin_translation":[0, 0, 0.04+0.004], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.082,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.03],
        "joint_origin_rpy":[0, 1.57-0.1, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 0, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_4_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.007,
        "geometry_length": 0.05,
        "joint_parent": "link_4_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.082+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_4_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.032,
        "joint_parent": "link_4_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.05+0.007], # parent's radius and length
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
        "joint_origin_translation":[0, 0, 0.032+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.08,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0.015, 0, 0.04],
        "joint_origin_rpy":[0, 1.57-0.2, 0],  # 1.57+x
        "joint_limit": {"lower": 0, "upper": 0, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_5_1",
        "geometry_type": "capsule",
        "geometry_radius": 0.006,
        "geometry_length": 0.04,
        "joint_parent": "link_5_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.08+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 10.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_2",
        "geometry_type": "capsule",
        "geometry_radius": 0.005,
        "geometry_length": 0.03,
        "joint_parent": "link_5_1",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.04+0.006], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
    {
        "name_code": "link_5_3",
        "geometry_type": "capsule",
        "geometry_radius": 0.0022,
        "geometry_length": 0.015,
        "joint_parent": "link_5_2",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation":[0, 0, 0.03+0.005], # parent's radius and length
        "joint_origin_rpy":[0, 0, 0],
        "joint_limit": {"lower": 0, "upper": 1.57, "effort": 5.0, "velocity": 1.5},
    },
]

# 创建 Agent 实例
initial_agent_hand = UrdfAgent(
    agent_code="gorilla_agent_hand",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
).to_dict()

if __name__ == "__main__":
    # 查看生成的 agent 数据
    #print(agent_demo.to_dict())

    # 调用函数
    generate_urdf_from_dict(initial_agent_hand, output_dir="output_meshes", output_urdf="gorilla_hand.urdf")
