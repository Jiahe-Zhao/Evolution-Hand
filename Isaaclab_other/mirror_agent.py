from class_agent import UrdfAgent
import copy
import uuid

def create_mirror_hand(initial_agent_hand,new_agent_name,mirror_axis='x'):
    mirrored = copy.deepcopy(initial_agent_hand)

     # 修改标识信息
    mirrored["agent_code"] = new_agent_name
    mirrored["evolution_id"] = uuid.uuid4().hex

    # 镜像处理所有链接
    for link in mirrored["links"]:
        # 坐标变换
        trans = link["joint_origin_translation"]
        if mirror_axis == 'x':
            link["joint_origin_translation"] = [-trans  [0], trans  [1], trans  [2]]
        # 旋转修正
        rpy = link["joint_origin_rpy"]
        if mirror_axis == 'x':
            link["joint_origin_rpy"] = [rpy  [0], -rpy  [1], -rpy  [2]]
            
        # 关节轴修正
        axis = link["joint_axis"]
        link["joint_axis"] = [axis  [0], axis  [1], axis  [2]] if mirror_axis == 'x' else axis
    return mirrored