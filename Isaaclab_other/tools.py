import random
import uuid
import copy
import math

def _parse_link_code(name_code):
    if not isinstance(name_code, str):
        return None
    if not name_code.startswith("link_"):
        return None
    parts = name_code.split("_")
    if len(parts) != 3:
        return None
    try:
        finger_id = int(parts[1])
        segment_id = int(parts[2])
    except ValueError:
        return None
    return finger_id, segment_id


def _group_links_by_finger(agent_data):
    groups = {}
    for link in agent_data.get("links", []):
        parsed = _parse_link_code(link.get("name_code", ""))
        if not parsed:
            continue
        finger_id, segment_id = parsed
        if finger_id == 0:
            continue
        groups.setdefault(finger_id, []).append((segment_id, link))

    for finger_id, items in groups.items():
        items.sort(key=lambda x: x[0])
        groups[finger_id] = [link for _, link in items]
    return groups


def _guess_thumb_finger_id(groups):
    if not groups:
        return None
    return min(groups.items(), key=lambda item: (len(item[1]), item[0]))[0]


def extract_name_codes(data, target_code):
    """
    从字典中提取某个字段
    :param data:
    :return:
    """
    name_codes = []
    # 如果是列表，则对每个元素递归调用
    if isinstance(data, list):
        for item in data:
            name_codes.extend(extract_name_codes(item, target_code))

    # 如果是字典，则检查每个键值对
    elif isinstance(data, dict):
        for key, value in data.items():
            if key == target_code:
                name_codes.append(value)
            else:
                name_codes.extend(extract_name_codes(value, target_code))
    return name_codes


def _shift_direct_child_joint_origins(agent_data, parent_name, delta_z):
    """Keep child joints at the distal end after parent geometry changes."""
    if not math.isfinite(float(delta_z)) or abs(float(delta_z)) < 1.0e-12:
        return
    for child in agent_data.get("links", []):
        if child.get("joint_parent") != parent_name:
            continue
        translation = list(child.get("joint_origin_translation", [0.0, 0.0, 0.0]))
        if len(translation) != 3:
            continue
        # Phalanges are generated along the parent's local +Z axis.
        translation[2] = float(translation[2]) + float(delta_z)
        child["joint_origin_translation"] = translation


def synchronize_kinematic_connections(agent_data):
    """Re-seat generated phalanges at the distal end of their parent.

    The source morphology stores each phalanx in its parent joint frame.  A
    length or radius mutation changes the distal attachment point, so every
    direct child in the same digit must follow that change.  Preserve lateral
    offsets, but make the longitudinal connection deterministic.
    """
    by_name = {
        link.get("name_code"): link
        for link in agent_data.get("links", [])
        if link.get("name_code")
    }
    for child in agent_data.get("links", []):
        child_name = child.get("name_code", "")
        parent_name = child.get("joint_parent")
        child_code = _parse_link_code(child_name)
        parent_code = _parse_link_code(parent_name)
        if not child_code or not parent_code or child_code[0] != parent_code[0]:
            continue
        if child_code[1] != parent_code[1] + 1 or parent_name not in by_name:
            continue
        parent = by_name[parent_name]
        try:
            distal_z = float(parent["geometry_length"]) + float(parent["geometry_radius"])
        except (KeyError, TypeError, ValueError):
            continue
        translation = list(child.get("joint_origin_translation", [0.0, 0.0, 0.0]))
        if len(translation) != 3:
            translation = [0.0, 0.0, 0.0]
        translation[2] = distal_z
        child["joint_origin_translation"] = translation
    return agent_data


def change_link_length(agent_data, link_name, step_length):
    """
    改变某一个 link 的长度，有 50% 的概率伸长，50% 的概率缩短
    :param agent_data: 字典数据，包含 link 信息
    :param link_name: 要修改的 link 名称
    :param step_length: 调整的比例（例如 0.1 表示调整 10%）
    :return: 修改后的 agent_data 或错误信息
    """
    for link in agent_data["links"]:
        if link.get("name_code") == link_name:
            # 随机选择伸长（+step_length）或缩短（-step_length）
            old_length = float(link["geometry_length"])
            adjustment = 1 + step_length * random.choice([1, -1])
            new_length = old_length * adjustment
            if new_length <= 0 or not math.isfinite(new_length):
                return f"Invalid mutated length for {link_name}."
            link["geometry_length"] = new_length
            _shift_direct_child_joint_origins(agent_data, link_name, new_length - old_length)
            return agent_data
    return f"Link {link_name} not found."


def change_link_radius(agent_data, link_name, step_length):
    """
    改变某一个 link 的长度，有 50% 的概率伸长，50% 的概率缩短
    :param agent_data: 字典数据，包含 link 信息
    :param link_name: 要修改的 link 名称
    :param step_length: 调整的比例（例如 0.1 表示调整 10%）
    :return: 修改后的 agent_data 或错误信息
    """
    for link in agent_data["links"]:
        if link.get("name_code") == link_name:
            # 随机选择伸长（+step_length）或缩短（-step_length）
            old_radius = float(link["geometry_radius"])
            adjustment = 1 + step_length * random.choice([1, -1])
            new_radius = old_radius * adjustment
            if new_radius <= 0 or not math.isfinite(new_radius):
                return f"Invalid mutated radius for {link_name}."
            link["geometry_radius"] = new_radius
            _shift_direct_child_joint_origins(agent_data, link_name, new_radius - old_radius)
            return agent_data
    return f"Link {link_name} not found."


def change_joint_origin_translation(agent_data, joint_name, step_length):
    """
    随机改变关节 joint_origin_translation 在三个维度上的位置
    :param agent_data: 字典数据，包含关节信息
    :param joint_name: 要修改的关节名
    :param step_length: 每个维度移动的最大步长
    :return: 修改后的 agent_data 或错误信息
    """
    for link in agent_data["links"]:
        if link.get("joint_name") == joint_name or link.get("name_code") == joint_name:
            # 获取当前的 joint_origin_translation，如果没有则初始化为 [0, 0, 0]
            current_translation = link.get("joint_origin_translation", [0, 0, 0])
            # 随机在每个维度上移动
            new_translation = [
                coord + step_length * random.choice([1, -1]) for coord in current_translation
            ]
            # 更新 joint_origin_translation
            link["joint_origin_translation"] = new_translation
            return agent_data
    return f"Joint {joint_name} not found."


def change_joint_origin_rpy(agent_data, joint_name, step_length):
    """
    随机改变关节 joint_origin_rpy 在三个欧拉角维度上的位置
    :param agent_data: 字典数据，包含关节信息
    :param joint_name: 要修改的关节名
    :param step_length: 每个维度最大移动步长（单位：弧度）
    :return: 修改后的 agent_data 或错误信息
    """
    for link in agent_data["links"]:
        if link.get("joint_name") == joint_name or link.get("name_code") == joint_name:
            name_code = link.get("name_code", "")
            parts = name_code.split("_")
            # Origin rotations on flexion-chain links change the meaning of
            # positive joint angle and can create a backward-bending finger.
            # Geometry/length evolution remains available, but this frame is
            # an anatomical invariant.
            if (
                len(parts) == 3
                and parts[0] == "link"
                and parts[1].isdigit()
                and 1 <= int(parts[1]) <= 5
                and parts[2].isdigit()
                and int(parts[2]) <= 2
            ):
                return agent_data
            # 获取当前的 joint_origin_rpy，如果没有则初始化为 [0, 0, 0]
            current_rpy = link.get("joint_origin_rpy", [0, 0, 0])
            # 随机在每个维度上移动
            new_rpy = [
                angle + step_length * random.choice([1, -1]) for angle in current_rpy
            ]
            # 更新 joint_origin_rpy
            link["joint_origin_rpy"] = new_rpy
            return agent_data
    return f"Joint {joint_name} not found."

def change_finger_length(agent_data, finger_id=None, step_length=0.1):
    """
    改变某个手指的整体长度（所有指节等比例缩放）
    :param agent_data: 字典数据，包含 link 信息
    :param finger_id: 手指编号（link_{finger_id}_*），None 表示自动推断拇指
    :param step_length: 调整比例（例如 0.1 表示调整 10%）
    :return: 修改后的 agent_data 或错误信息
    """
    groups = _group_links_by_finger(agent_data)
    if finger_id is None:
        finger_id = _guess_thumb_finger_id(groups)
    if finger_id not in groups:
        return f"Finger {finger_id} not found."

    adjustment = 1 + step_length * random.choice([1, -1])
    for link in groups[finger_id]:
        if "geometry_length" in link:
            old_length = float(link["geometry_length"])
            new_length = old_length * adjustment
            if new_length <= 0 or not math.isfinite(new_length):
                return f"Invalid mutated length for {link.get('name_code', '<unnamed>')}."
            link["geometry_length"] = new_length
            _shift_direct_child_joint_origins(
                agent_data, link.get("name_code"), new_length - old_length
            )
    return agent_data


def change_thumb_length(agent_data, step_length=0.1):
    """
    改变拇指长度（自动推断拇指为指节数量最少的手指）
    """
    return change_finger_length(agent_data, finger_id=None, step_length=step_length)


def change_palm_curvature(agent_data, step_angle, axis="y"):
    """
    改变掌心曲率：调整所有掌根关节的 joint_origin_rpy
    :param agent_data: 字典数据，包含 link 信息
    :param step_angle: 每次调整的最大角度（弧度）
    :param axis: 'x'/'y'/'z' 选择旋转轴
    :return: 修改后的 agent_data 或错误信息
    """
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_index is None:
        return f"Invalid axis {axis}."

    base_names = {link["name_code"] for link in agent_data.get("base_link", []) if "name_code" in link}
    if not base_names:
        return "Base link not found."

    affected = 0
    for link in agent_data.get("links", []):
        if link.get("joint_parent") in base_names:
            current_rpy = link.get("joint_origin_rpy", [0, 0, 0])
            if len(current_rpy) < 3:
                current_rpy = list(current_rpy) + [0] * (3 - len(current_rpy))
            current_rpy[axis_index] = current_rpy[axis_index] + step_angle * random.choice([1, -1])
            link["joint_origin_rpy"] = current_rpy
            affected += 1

    if affected == 0:
        return "No palm joints found."
    return agent_data


def remove_link(agent_data, link_name):
    """
    删除某一个 link 及其所有子 link
    :param agent_data: 包含 link 数据的字典
    :param link_name: 要删除的 link 的 name_code
    :return: 修改后的 agent_data 或错误信息
    """
    # The thumb must retain at least the human baseline articulation.  Its
    # topology is protected here so evolutionary deletion cannot remove the
    # opposition/flexion chain before URDF generation.
    if str(link_name).startswith("link_1_"):
        return f"Protected thumb link cannot be removed: {link_name}."

    parsed = _parse_link_code(link_name)
    if parsed and parsed[1] == 0:
        return f"Protected digit root cannot be removed: {link_name}."

    # 创建子链路映射 {parent_name: [child_links]}
    parent_to_children = {}
    for link in agent_data["links"]:
        parent = link.get("joint_parent")
        if parent:
            parent_to_children.setdefault(parent, []).append(link["name_code"])

    # 获取需要删除的所有链接（递归获取子链接）
    links_to_remove = set()

    def collect_links_to_remove(parent):
        """递归收集所有需要删除的子链接"""
        if parent in links_to_remove:
            return
        links_to_remove.add(parent)
        for child in parent_to_children.get(parent, []):
            collect_links_to_remove(child)

    # 开始删除操作
    collect_links_to_remove(link_name)

    # 更新 links 列表
    initial_count = len(agent_data["links"])
    agent_data["links"] = [
        link for link in agent_data["links"] if link.get("name_code") not in links_to_remove
    ]

    if len(agent_data["links"]) < initial_count:
        return agent_data
    return f"Link {link_name} not found."





def add_link(agent_data, parent_link_name):
    """
    添加一个新链接到没有子链接的目标链接上
    :param agent_data: 包含 link 数据的字典
    :param parent_link_name: 目标父链接的 name_code
    :return: 修改后的 agent_data 或错误信息
    """
    # 创建子链路映射 {parent_name: [child_links]}
    parent_to_children = {}
    for link in agent_data["links"]:
        parent = link.get("joint_parent")
        if parent:
            parent_to_children.setdefault(parent, []).append(link["name_code"])

    # 确保目标链接没有子链接
    if parent_link_name in parent_to_children:
        return f"Link {parent_link_name} already has child links, cannot add."

    # 查找目标链接
    parent_link = next((link for link in agent_data["links"] if link.get("name_code") == parent_link_name), None)
    if not parent_link:
        return f"Parent link {parent_link_name} not found."

    # Keep the canonical numeric chain naming.  Downstream URDF and task
    # adapters use this naming to discover the actual fingertip after adding
    # or removing distal phalanges.
    new_link = copy.deepcopy(parent_link)
    parsed_parent = _parse_link_code(parent_link_name)
    if not parsed_parent:
        return f"Cannot add a phalanx to non-canonical link: {parent_link_name}."
    finger_id, parent_segment = parsed_parent
    existing_segments = []
    for link in agent_data.get("links", []):
        parsed = _parse_link_code(link.get("name_code", ""))
        if parsed and parsed[0] == finger_id:
            existing_segments.append(parsed[1])
    new_segment = max(existing_segments + [parent_segment]) + 1
    new_name_code = f"link_{finger_id}_{new_segment}"
    if any(link.get("name_code") == new_name_code for link in agent_data["links"]):
        return f"Link {new_name_code} already exists."
    new_link["name_code"] = new_name_code
    new_link["joint_name"] = f"{parent_link_name}_to_{new_name_code}"
    new_link["joint_parent"] = parent_link_name
    # Attach at the distal end, not at the parent's origin. Otherwise two
    # coincident solids are hidden by the adjacent-joint collision filter.
    length = float(parent_link.get("geometry_length", 0.0))
    radius = float(parent_link.get("geometry_radius", 0.0))
    if not math.isfinite(length) or not math.isfinite(radius) or length <= 0 or radius <= 0:
        return f"Invalid parent geometry for new phalanx: {parent_link_name}."
    new_link["joint_origin_translation"] = [0.0, 0.0, length + radius]
    new_link["joint_origin_rpy"] = [0, 0, 0]          # 默认 joint rotation
    new_link["joint_type"] = "revolute"
    new_link["joint_axis"] = [1, 0, 0]
    new_link["joint_limit"] = dict(new_link.get("joint_limit", {}))
    new_link["joint_limit"]["lower"] = 0.0
    new_link["joint_limit"]["upper"] = max(0.1, float(new_link["joint_limit"].get("upper", 1.57)))

    # 添加新链接到 agent_data
    agent_data["links"].append(new_link)

    synchronize_kinematic_connections(agent_data)

    return agent_data
