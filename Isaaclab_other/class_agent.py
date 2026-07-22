import uuid

class UrdfAgent:
    def __init__(self, agent_code, base_link_name, base_link_geometry, links=[]):
        """
        初始化Agent类

        :param agent_code: 机器人的代码名称
        :param base_link_name: 基础链接的名称
        :param base_link_geometry: 基础链接的几何形状参数 (dict)
        :param links: 链接参数列表，每个元素是一个 dict，指定单独的 link 和 joint 配置
        """
        # 这里定义了agent类所包含的结构
        self.agent = {
            "agent_code": agent_code,
            "base_link": [
                {
                    "name_code": base_link_name,
                    **base_link_geometry,
                }
            ],
            "links": [],
            "evolution_information": [],
            "evolution_id" : uuid.uuid4().hex,  # 唯一的 ID
        }

        # 自动添加初始 links
        for link in links:
            self.add_link(link)

    def add_link(self, link_config):
        """
        添加一个新链接和对应的关节。

        :param link_config: 包含 link 和 joint 配置的 dict，例如：
            {
                "name_code": "link_1_0",
                "geometry_type": "capsule",
                "geometry_radius": 0.1,
                "geometry_length": 0.4,
                "joint_parent": "link_0_0",
                "joint_type": "revolute",
                "joint_axis": [0, 0, 1],
                "joint_limit": {"lower": -1.57, "upper": 1.57, "effort": 10.0, "velocity": 1.0},
                "joint_origin_translation": [0, 0, 0.2],
                "joint_origin_rpy": [0, 0, 0],
            }
        """
        joint_parent = link_config.get("joint_parent", "base_link")
        joint_name = f"{joint_parent}_to_{link_config.get('name_code', 'new_link')}"  # 根据父链接和当前链接名动态生成

        default_joint_config = {
            "joint_name": joint_name,
            "joint_type": "revolute",
            "joint_axis": [0, 0, 1],
            "joint_limit": {"lower": -1.57, "upper": 1.57, "effort": 10.0, "velocity": 1.0},
            "joint_origin_translation": [0, 0, 0],
            "joint_origin_rpy": [0, 0, 0],
        }

        # 合并默认 joint 配置与用户自定义配置
        joint_config = {**default_joint_config, **link_config}

        # 将 link 配置和对应的 joint 添加到列表
        self.agent["links"].append({
            "name_code": link_config["name_code"],
            "geometry_type": link_config.get("geometry_type", "capsule"),
            "geometry_radius": link_config.get("geometry_radius", 0.1),
            "geometry_length": link_config.get("geometry_length", 0.4),
            "geometry_size": link_config.get("geometry_size"),
            "joint_name": joint_config["joint_name"],
            "joint_parent": joint_parent,
            "joint_type": joint_config["joint_type"],
            "joint_axis": joint_config["joint_axis"],
            "joint_limit": joint_config["joint_limit"],
            "joint_origin_translation": joint_config["joint_origin_translation"],
            "joint_origin_rpy": joint_config["joint_origin_rpy"],
        })

    def update_link_geometry(self, link_name, new_geometry):
        """
        更新指定链接的几何形状。

        :param link_name: 要更新的链接名称
        :param new_geometry: 新几何形状参数 (dict)
        """
        for link in self.agent["links"]:
            if link["name_code"] == link_name:
                link.update(new_geometry)
                return f"Geometry of {link_name} updated."
        return f"Link {link_name} not found."

    def update_joint_parameters(self, link_name, new_joint_params):
        """
        更新指定链接的关节参数。

        :param link_name: 要更新的链接名称
        :param new_joint_params: 新的关节参数 (dict)
        """
        for link in self.agent["links"]:
            if link["name_code"] == link_name:
                link.update(new_joint_params)
                return f"Joint parameters of {link_name} updated."
        return f"Link {link_name} not found."

    def to_dict(self):
        """
        获取当前 agent 数据字典。
        """
        return self.agent


'''
# 示例用法
# 定义基础链接的几何形状
base_geometry = {
    "geometry_type": "capsule",
    "geometry_radius": 0.15,
    "geometry_length": 0.6,
}

# 定义多个链接和关节的参数
links = [
    {
        "name_code": "link_1_0",
        "geometry_type": "capsule",
        "geometry_radius": 0.1,
        "geometry_length": 0.4,
        "joint_parent": "link_0_0",
        "joint_axis": [1, 0, 0],
        "joint_origin_translation": [0, 0, 0.2],
        "joint_origin_rpy": [0, 0, 0],
        "joint_limit": {"lower": -0.5, "upper": 0.5, "effort": 15.0, "velocity": 2.0},
    },
    {
        "name_code": "link_1_1",
        "geometry_type": "box",
        "geometry_radius": None,
        "geometry_length": 0.2,
        "joint_parent": "link_1_0",
        "joint_origin_translation": [0.1, 0, 0],
        "joint_origin_rpy": [0, 0, 0],
    },
]

# 创建 Agent 实例
agent_demo = UrdfAgent(
    agent_code="custom_robot",
    base_link_name="link_0_0",
    base_link_geometry=base_geometry,
    links=links
)

# 添加一个新链接
agent_demo.add_link({
    "name_code": "link_1_3",
    "geometry_type": "capsule",
    "geometry_radius": 0.2,
    "geometry_length": 0.6,
    "joint_parent": "link_1_2",
    "joint_origin_translation": [0, 0, 0.5],
    "joint_origin_rpy": [0, 1.57, 0],
    "joint_limit": {"lower": -0.8, "upper": 0.8, "effort": 20.0, "velocity": 3.0},
})
print(agent_demo.update_link_geometry("link_1_3",{'geometry_type': 'capsule', 'geometry_radius': 0.1, 'geometry_length': 0.4}))
# 查看生成的 agent 数据
print(agent_demo.to_dict())
'''
