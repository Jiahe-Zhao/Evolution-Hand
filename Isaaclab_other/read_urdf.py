import open3d as o3d
from urdfpy import URDF
import numpy as np

# 加载 URDF 文件
robot = URDF.load("human_hand.urdf")
#robot = URDF.load("gorilla_hand.urdf")

# 保存所有几何信息
geometries = []

# 处理每个链接
for link in robot.links:
    print(link.name)

for joint in robot.joints:
    print('{} connects {} to {}'.format( joint.name, joint.parent, joint.child))

for joint in robot.actuated_joints:
    print(joint.name)

print(robot.base_link.name)

fk = robot.link_fk()
print(fk[robot.links[0]])

print(fk[robot.links[1]])

fk = robot.link_fk(cfg={'link_0_0_to_link_1_0' : 1.0})
print(fk[robot.links[1]])


fk = robot.visual_trimesh_fk()
print(type(list(fk.keys())[0]))

#fk = robot.collision_trimesh_fk()
#print(type(list(fk.keys())[0]))

#robot.show(cfg={'joint1': 2})

robot.animate(cfg_trajectory={'link_2_0_to_link_2_1' : [0, np.pi / 4],})