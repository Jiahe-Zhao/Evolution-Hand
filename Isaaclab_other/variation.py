import random
from tools import *
import numpy as np
"""
def(对象,变异类型）
然后再是子函数
包括
    改变 link 长度
    改变 link 半径
    减少link
    改变关节 orgin
    改变关节 角度范围
    新增 link （复制原有手指的1或2或3个指节与joint）

不要每次都进行重命名

"""

TASK_SPECS = [
    ("change_link_length", "link"),
    ("change_link_radius", "link"),
    ("change_joint_origin_translation", "link"),
    ("change_joint_origin_rpy", "link"),
    ("change_thumb_length", "global"),
    ("change_palm_curvature", "global"),
]


def _build_task_candidates(robot_data, task_specs=None):
    if task_specs is None:
        task_specs = TASK_SPECS
    all_name_codes = extract_name_codes(robot_data, "name_code")
    all_name_codes = [code for code in all_name_codes if code != "link_0_0"]

    candidates = []
    for task, target_type in task_specs:
        if target_type == "link":
            candidates.extend((task, link) for link in all_name_codes)
        elif target_type == "global":
            # 让全局任务与 link 任务有相近的采样概率
            repeat = max(1, len(all_name_codes))
            for _ in range(repeat):
                candidates.append((task, None))
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
    return candidates


def check_success(robot_data):
    """
    验证机器人结构的物理约束和拓扑完整性
    返回布尔值表示是否通过所有检查
    """
    # 基础数据结构校验
    if not validate_structure(robot_data):
        print("1")
        return False
    
    # 物理参数校验
    if not check_physical_params(robot_data):
        print("2")
        return False
    
    # 关节连接性校验
    if not check_joint_connections(robot_data):
        print("3")
        return False
    
    # 拓扑连通性校验
    if not check_topology_integrity(robot_data):
        print("4")
        return False
    
    # 关节位移合理性校验
    if not check_joint_displacements(robot_data):
        print("5")
        return False
    
    return True

# 辅助函数 ------------------------------------------------------------

def validate_structure(robot_data):
    """基础数据结构验证"""
    # print("robot_data:",robot_data)
    required_keys = {"links", "base_link"}
    if not required_keys.issubset(robot_data.keys()):
        print("[Error] 缺少关键字段: base_link 或 links")
        return False
    

    
    for link in robot_data["links"]:
        if "name_code" not in link:
            print(f"[Error] 存在未命名的链接: {link}")
            return False
        if "joint_parent" in link and link["joint_parent"] is None:
            if link["name_code"] != robot_data["base_link"]:
                print(f"[Error] 非基链接缺少父关节: {link['name_code']}")
                return False
    return True

    
def check_physical_params(robot_data):
    """几何参数合理性检查"""
    for link in robot_data["links"]:
        # 长度必须为正
        if "geometry_length" in link:
            if float(link["geometry_length"]) <= 0:
                print(f"[Error] 非法长度: {link['name_code']} {link['geometry_length']}")
                return False
        
        # 半径必须为正
        if "geometry_radius" in link:
            if float(link["geometry_radius"]) <= 0:
                print(f"[Error] 非法半径: {link['name_code']} {link['geometry_radius']}")
                return False
        
        # 质量属性检查
        if "mass" in link:
            if float(link["mass"]) <= 0:
                print(f"[Error] 非法质量值: {link['name_code']} {link['mass']}")
                return False
    return True

def check_joint_connections(robot_data):
    """关节连接性检查"""
    
  
    
    # 提取所有基链接的name_code
    base_link_names = {link["name_code"] for link in robot_data["base_link"]}  # 使用集合提高查找效率
    
    # 收集所有存在的链接名称（包括基链接）
    all_link_names = base_link_names.union(
        {link["name_code"] for link in robot_data["links"]}
    )
    for link in robot_data["links"]:

        # 检查非基链接的父链接有效性
        if "joint_parent" in link:
            parent = link["joint_parent"]
            if parent is None:
                # 如果是基链接，应存在于base_link中
                if link["name_code"] not in base_link_names:
                    print(f"[Error] 未声明的基链接: {link['name_code']}")
                    return False
            else:
                # 检查父链接是否存在
                
                if parent not in all_link_names:
                    print("link:",link)
                    print(f"[Error] 父链接不存在: {link['name_code']} -> {parent}")
                    return False

   
    
    # 检查循环依赖
    visited = set()
    parent_map = {}
    for link_name in parent_map:
        current = link_name
        while current in parent_map:
            if current in visited:
                break
            visited.add(current)
            current = parent_map[current]
            if current == link_name:
                print(f"[Error] 循环依赖: {link_name} -> {current}")
                return False
    return True

def check_topology_integrity(robot_data):
    """拓扑连通性检查（BFS验证）"""
    from collections import deque
    # 获取所有基链接的名称列表
    base_names = [link["name_code"] for link in robot_data["base_link"]]
    # 构建邻接表
    adjacency = {}
    for link in robot_data["links"]:
        # if "joint_parent" in link:
        #     parent = link["joint_parent"]
        #     if parent:
        #         adjacency.setdefault(parent, []).append(link["name_code"])
        if "name_code" not in link:
            print("[Error] 存在未命名的链接")
            return False
        if "joint_parent" not in link:
            print(f"[Error] 链接 {link['name_code']} 缺少 joint_parent 字段")
            return False
            
        parent = link["joint_parent"]
        child = link["name_code"]
        
        # 处理父子关系（允许基链接的 joint_parent 为空）
        if parent:
            adjacency.setdefault(parent, []).append(child)

    # 从基链接开始遍历
    # base = robot_data["base_link"]
    visited = set()
    queue = deque(base_names)
    
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current in adjacency:
            queue.extend(adjacency[current])
    
    # 验证所有链接都被访问
    total_links = {link["name_code"] for link in robot_data["links"]}
    missing_links = total_links - visited
    
    if missing_links:
        print(f"[Error] 存在 {len(missing_links)} 个孤立链接: {missing_links}")
        return False
    return True

def check_joint_displacements(robot_data, max_distance=1.0):
    """关节位移合理性检查（单位：米）"""
    # 构建位置索引
    positions = {}
    for link in robot_data["links"]:
        if "joint_origin_translation" in link:
            positions[link["name_code"]] = np.array(
                list(map(float, link["joint_origin_translation"]))
            )
    
    for link in robot_data["links"]:
        if "joint_parent" not in link:
            continue
            
        parent = link["joint_parent"]
        if not parent:
            continue
            
        if link["name_code"] not in positions or parent not in positions:
            continue
            
        # 计算相对位移
        child_pos = positions[link["name_code"]]
        parent_pos = positions[parent]
        distance = np.linalg.norm(child_pos - parent_pos)
        
        if distance > max_distance:
            print(f"[Warning] 关节位移过大: {parent} -> {link['name_code']} ({distance:.2f}m)")
            return False
    return True


def choose_target_reward(robot_data, mutation_stats=None, epsilon=0.2):
    all_combinations = _build_task_candidates(robot_data)
    # 进行 ε-greedy 探索：有 epsilon 的概率完全随机选择
    if mutation_stats and random.random() > epsilon:
        # 用历史的 score_delta 来加权选择
        weights = []
        for task, link in all_combinations:
            key = (task, link)
            stat = mutation_stats.get(key, {"trials": 0, "improvements": 0.0, "score_delta": 0.0})
            # 使用 score_delta 作为权重（较大的正向变化给更高权重）
            trials=stat["trials"]
            improvements=stat["improvements"]
            delta = stat["score_delta"]  # 更高的 score_delta 表示更有价值

            trials += 1       # 平滑，避免除以 0
            improvements += 1

            avg_delta = delta / trials
            success_rate = improvements / trials
            weights.append(avg_delta * success_rate)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            probabilities = [w / total_weight for w in weights]
        else:
            # 所有 score_delta 都为 0，使用均匀分布
            # print("[Warning] All score_delta are zero — falling back to uniform probability.")
            probabilities = [1.0 / len(weights)] * len(weights)

        # 根据权重选择 (task, link) 组合
        selected_task, selected_link = random.choices(all_combinations, weights=probabilities, k=1)[0]
    else:
        # 完全随机选择任务和链接组合
        selected_task, selected_link = random.choice(all_combinations)

    # 生成变异强度（随机值）
    random_alpha = random.random()
    return selected_link, selected_task, random_alpha
    


def choose_target(robot_data, task_probabilities=None):
    """
    选取变异对象：先选 name_code, 再选变异任务，随机生成变异系数（大变异与小变异），
    同时允许指定每个变异任务的发生概率。

    :param robot_data: 包含机器人数据的字典
    :param task_probabilities: 一个字典，指定变异任务及其发生概率，例如：
                               {"change_link_length": 0.2, "change_link_radius": 0.3, ...}
    :return: 选定的 name_code, 变异任务, 变异系数

    例如
    task_probabilities = {
    "change_link_length": 0.5,
    "change_link_radius": 0.2,
    "remove_link": 0.1,
    "add_link": 0.1,
    "change_joint_origin_translation": 0.05,
    "change_joint_origin_rpy": 0.05,}
    """
    all_name_codes = extract_name_codes(robot_data, "name_code")

    # 如果存在 name_code，则随机选择一个
    if all_name_codes:
        random_name_code = random.choice(all_name_codes)
        print(f"Randomly selected name_code: {random_name_code}")
    else:
        raise ValueError("No name_code found in the provided data.")

    # 变异任务列表及默认概率
    task_list = [
        "change_link_length",
        "change_link_radius",
        "remove_link",
        "add_link",
        "change_joint_origin_translation",
        "change_joint_origin_rpy",
        "change_thumb_length",
        "change_palm_curvature",
    ]
    default_probabilities = {task: 1 / len(task_list) for task in task_list}

    # 如果用户指定了概率，则更新默认概率
    if task_probabilities:
        for task, prob in task_probabilities.items():
            if task in default_probabilities:
                default_probabilities[task] = prob
            else:
                raise ValueError(f"Task {task} is not a valid task.")

    # 确保概率总和为1（可自动归一化）
    total_prob = sum(default_probabilities.values())
    normalized_probabilities = [default_probabilities[task] / total_prob for task in task_list]

    # 按概率随机选择变异任务
    random_task = random.choices(task_list, weights=normalized_probabilities, k=1)[0]

    # 生成变异系数
    random_alpha = random.random()

    return random_name_code, random_task, random_alpha

import copy

def variation(robot_data, random_name_code, random_task, random_alpha, standard_variation = 0.1, standard_length = 0.05):
    """
    根据选择的对象，进行变异动作
    :param robot_data: 变异前的定义字典
    :param random_name_code: 变异对象（link的名称）
    :param random_task: 变异任务
    :param random_alpha: 变异比例从0-1选
    :param standard_variation: 标准的变异比例为10%
    :return: 变异过后的定义字典
    """
    urdf_copy = copy.deepcopy(robot_data)
    if random_task == "change_link_length":
        result = change_link_length(urdf_copy, random_name_code, random_alpha * standard_variation)
        if isinstance(result, str):  # 返回字符串代表失败
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "change_link_radius":
        result = change_link_radius(urdf_copy, random_name_code, random_alpha * standard_variation)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "remove_link":
        result = remove_link(urdf_copy, random_name_code)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "add_link":
        result = add_link(urdf_copy, random_name_code)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "change_joint_origin_translation":
        result = change_joint_origin_translation(urdf_copy, random_name_code, random_alpha * standard_length)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "change_joint_origin_rpy":
        result = change_joint_origin_rpy(urdf_copy, random_name_code, random_alpha * standard_length)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "change_thumb_length":
        result = change_thumb_length(urdf_copy, random_alpha * standard_variation)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    elif random_task == "change_palm_curvature":
        result = change_palm_curvature(urdf_copy, random_alpha * standard_length)
        if isinstance(result, str):
            task_status = "task_failed"
        else:
            urdf_copy = result
            task_status = "task_successful"

    if task_status == "task_failed":
        success = False
    else:
        success = check_success(urdf_copy)  # 要确保移动之后符合物理约束，比如origin不能离父节点太远

    return success, urdf_copy


def seed_initial_population(
    base_urdf,
    population_size,
    include_base=True,
    max_attempts=200,
    standard_variation=0.05,
    standard_length=0.02,
):
    """
    通过轻微变异生成初始种群
    """
    population = []
    if include_base:
        population.append(base_urdf)

    attempts = 0
    while len(population) < population_size and attempts < max_attempts:
        link_code, task_code, strength = choose_target_reward(base_urdf, mutation_stats=None, epsilon=1.0)
        success_tag, new_urdf = variation(
            base_urdf,
            link_code,
            task_code,
            strength,
            standard_variation=standard_variation,
            standard_length=standard_length,
        )
        if success_tag:
            population.append(new_urdf)
        attempts += 1

    return population




