import xml.etree.ElementTree as ET
import os


def parse_urdf_and_generate_articulation_cfg(urdf_path, asset_path, output_py_file):
    """
    根据 URDF 文件的内容生成 ArticulationCfg 配置并保存为 .py 文件。

    :param urdf_path: (str) URDF 文件的路径
    :param asset_path: (str) 指定的资产路径
    :param output_py_file: (str) 输出的 Python 文件路径
    :return: None
    """

    # 解析 URDF 文件
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 提取链接和关节信息
    links = []
    joints = []
    effort_limits = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        links.append(link_name)

    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')  # Can be "revolute", "continuous", etc.
        if joint_type == "fixed":
            continue
        effort = joint.find('limit').get('effort') if joint.find('limit') is not None else None

        joints.append(joint_name)
        if effort:
            effort_limits[joint_name] = float(effort)

    # 动态生成 ArticulationCfg 配置 要改
    evolution_root = os.environ.get("EVOLUTION_ROOT", os.path.join(os.path.expanduser("~"), "Evolution"))
    if asset_path.startswith(evolution_root + os.sep):
        asset_rel_path = os.path.relpath(asset_path, evolution_root)
    else:
        asset_rel_path = asset_path

    articulation_cfg = f"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.converters import UrdfConverterCfg

EVOLUTION_ROOT = os.environ.get("EVOLUTION_ROOT", os.path.join(os.path.expanduser("~"), "Evolution"))
ASSET_PATH = os.path.join(EVOLUTION_ROOT, {asset_rel_path!r})

# Configuration based on URDF file {urdf_path}
CURRENT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ASSET_PATH,
        activate_contact_sensors=False,  # 禁用接触传感器模拟
        fix_base=True, #True
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # 禁用重力
            retain_accelerations=True,  # 保留加速度
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,  # 启用自碰撞
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(drive_type="force",
                                                   gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=30.1,damping=0.1)),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),  # 将驱动器类型设置为“强制”
        # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),  # 配置具有刚度和阻尼的肌腱
    ),
    init_state=ArticulationCfg.InitialStateCfg(  # 在这里修改初始化姿态并没有用
        pos=(0.0, 0.0, 0.5),  # 初始位置
        rot=(0.0, 0.0, -0.7071, 0.7071),  # 四元数表示的初始方向
        joint_pos={{".*": 0.0}},  # 初始状态关节位置全部设为0
    ),
    actuators={{
"""

    # 根据关节信息生成执行器配置
    actuator_cfgs = {}

    actuator_cfgs["hand"] = {
        "joint_names_expr": [],
        "effort_limit_sim": {},
        "stiffness": {},
        "damping": {},
    }

    for joint in joints:
        actuator_cfgs["hand"]["joint_names_expr"].append(joint)
        actuator_cfgs["hand"]["effort_limit_sim"][joint] = effort_limits.get(joint, 4.785)
        actuator_cfgs["hand"]["stiffness"][joint] = 1.0
        actuator_cfgs["hand"]["damping"][joint] = 0.1

    # Now generate the configuration for each actuator group (e.g., "fingers")
    for finger, cfg in actuator_cfgs.items():
        articulation_cfg += f"""
        "{finger}": ImplicitActuatorCfg(
            joint_names_expr={cfg["joint_names_expr"]},
            effort_limit_sim={cfg["effort_limit_sim"]},
            stiffness={cfg["stiffness"]},
            damping={cfg["damping"]},
        ),"""

    articulation_cfg += """
    },
    soft_joint_pos_limit_factor=1.0,
)
"""

    # 确保目录存在，如果目录不存在，则创建
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # 将生成的配置保存到 Python 文件
    with open(output_py_file, 'w') as f:
        f.write(articulation_cfg)

    print(f"ArticulationCfg has been saved to {output_py_file}.")


def calculate_observation_number (urdf_path):
    """
    calculate observation number for different agent structure
    :param urdf_path:
    :return:
    """


    observation_number = 157 #需要改
    return observation_number

def task_generate_env_cfg(current_task, urdf_path,mirror_urdf_path, observation_number, output_py_file):
    if current_task == 'Isaac-Hand-Cube-v0':
        cube_task_generate_env_cfg(urdf_path, observation_number, output_py_file)
    elif current_task =='Isaac-EvolutionHand-Grasp-v0':
        grasp_task_generate_env_cfg(urdf_path,mirror_urdf_path, observation_number, output_py_file) #生成evolution_stone_grind_env_cfg.py
    elif current_task =='Isaac-EvolutionHand-Manipulation-v0':
        manipulation_task_generate_env_cfg(urdf_path,mirror_urdf_path, observation_number, output_py_file) #生成evolution_stone_grind_env_cfg.py
    elif current_task =='Isaac-EvolutionHand-Strike-v0':
        strike_task_generate_env_cfg(urdf_path,mirror_urdf_path, observation_number, output_py_file) #生成evolution_stone_grind_env_cfg.py
    elif current_task =='Isaac-EvolutionHand-StoneGrind-v0':
        stonegrind_task_generate_env_cfg(urdf_path,mirror_urdf_path, observation_number, output_py_file) #生成evolution_stone_grind_env_cfg.py

def grasp_task_generate_env_cfg(urdf_path, mirror_urdf_path,observation_number, output_py_file):
    """
    Generate an environment configuration .py file based on the given URDF path and observation number.

    Args:
        urdf_path (str): Path to the URDF file.
        observation_number (int): Number of observations.
        output_py_file (str): Path to save the generated Python file.
    """

    # Helper function to parse URDF and extract joint and link names
    def parse_urdf(urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = []
        link_names = []

        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            if joint_name:
                joint_names.append(joint_name)

        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        return joint_names, link_names

    # Parse the URDF file
    left_joint_names, left_link_names = parse_urdf(urdf_path) #left hand

    #right hand =mirror hand
    right_joint_names, right_link_names = parse_urdf(mirror_urdf_path)


    # Create Python content for the environment configuration
    env_cfg_content = f"""
from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
import math

@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={{
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        }},
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        }},
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }},
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={{
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )

from collections import defaultdict
@configclass
class EvolutionGraspEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    # urdf_path = "{urdf_path}"
    

    # Actuated joints and fingertip links
    actuated_joint_names = {left_joint_names}
    finger_body_names = {left_link_names}

    #分离出指尖
    finger_links = defaultdict(list)
    # Step 1: 分组
    for name in finger_body_names:
        parts = name.split('_')
        if len(parts) != 3:
            continue
        finger_id = parts[1]
        joint_id = int(parts[2])
        # 跳过手掌 link_0_0
        if finger_id == '0':
            continue
        finger_links[finger_id].append((joint_id, name))

    # Step 2: 找最大 joint_id 的 link
    fingertip_body_names = []
    
    for finger_id, joints in finger_links.items():
        max_joint = max(joints, key=lambda x: x[0])
        fingertip_body_names.append(max_joint[1])  # only name
    
    # Environment settings
    decimation = 2
    episode_length_s = 10.0
    
    action_space = len(actuated_joint_names)
    
    observation_space = len(actuated_joint_names)*3+len(fingertip_body_names)*13+16
   
    # state_space = (len(actuated_joint_names)*3+len(fingertip_body_names)*19)+16
    state_space=0
    asymmetric_obs = False
    obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    
    # Robot configuration (using parsed joint names)
    #grasp_hand 位置还得改 左手
    robot_cfg: ArticulationCfg = LEFT_HAND_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(-0.707107, 0.707107, 0.0, 0),
            joint_pos={{".*": 0.0}},
        )
    )
    
    #grasp_object_cfg
    grasp_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.005,  # 可以尝试增加此值
                # rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.01, 0.378),  
            rot=(1.0,0.0,0.0,0.0),#初始状态 
        )
    )
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        # filter_prim_paths_expr=["/World/envs/env_.*/Cone"],
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=20, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    # fall_dist = 0.24
    transition_scale=0.5
    orientation_scale=0.5
    vel_obs_scale = 0.2
    act_moving_average = 1.0 #1.0 ???
    force_torque_obs_scale = 10.0
    # reward-related scales
    # dist_reward_scale = 20.0

    # reward scales
    dist_reward_scale = -1.0#-1.0
    angle_reward_scale=-3.0
    force_reward_scale= -5.0
    action_penalty_scale = -0.0002
    reach_goal_bonus = 1000
    fall_penalty = 0
    # grasp_dist = 0.025
    success_tolerance = 1
    max_consecutive_success = 0
    av_factor = 0.1
    fall_dist=0.15
"""

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # Write the generated content to the .py file
    with open(output_py_file, "w") as f:
        f.write(env_cfg_content)

    print(f"Environment configuration has been written to: {output_py_file}")

def manipulation_task_generate_env_cfg(urdf_path, mirror_urdf_path,observation_number, output_py_file):
    """
    Generate an environment configuration .py file based on the given URDF path and observation number.

    Args:
        urdf_path (str): Path to the URDF file.
        observation_number (int): Number of observations.
        output_py_file (str): Path to save the generated Python file.
    """

    # Helper function to parse URDF and extract joint and link names
    def parse_urdf(urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = []
        link_names = []

        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            if joint_name:
                joint_names.append(joint_name)

        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        return joint_names, link_names

    # Parse the URDF file
    left_joint_names, left_link_names = parse_urdf(urdf_path) #left hand

    #right hand =mirror hand
    right_joint_names, right_link_names = parse_urdf(mirror_urdf_path) #right hand


    # Create Python content for the environment configuration
    env_cfg_content = f"""
import os

from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg


@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={{
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        }},
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        }},
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }},
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={{
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )

from collections import defaultdict

HOME_DIR = os.path.expanduser("~")
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", os.path.join(HOME_DIR, "IsaacLab"))
EVOLUTION_TASKS_ROOT = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "evolution_tasks"
)
DEX_CUBE_USD_PATH = os.path.join(
    EVOLUTION_TASKS_ROOT, "task_manipulation", "DexCube", "dex_cube_new.usd"
)

@configclass
class EvolutionManipulationEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    # urdf_path = "{urdf_path}"
    # observation_number = {observation_number}

    # Actuated joints and fingertip links
    actuated_joint_names = {left_joint_names}
    finger_body_names = {left_link_names}

    #分离出指尖
    finger_links = defaultdict(list)
    # Step 1: 分组
    for name in finger_body_names:
        parts = name.split('_')
        if len(parts) != 3:
            continue
        finger_id = parts[1]
        joint_id = int(parts[2])
        # 跳过手掌 link_0_0
        if finger_id == '0':
            continue
        finger_links[finger_id].append((joint_id, name))

    # Step 2: 找最大 joint_id 的 link
    fingertip_body_names = []
    
    for finger_id, joints in finger_links.items():
        max_joint = max(joints, key=lambda x: x[0])
        fingertip_body_names.append(max_joint[1])  # only name

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = len(actuated_joint_names)
    observation_space = len(actuated_joint_names)*3+len(fingertip_body_names)*13+24  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    
    # Robot configuration (using parsed joint names)
    #grasp_hand 位置还得改 左手
    robot_cfg: ArticulationCfg = LEFT_HAND_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(-0.707107, 0.707107, 0.0, 0),
            joint_pos={{".*": 0.0}},
        )
    )
    
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            
            usd_path=DEX_CUBE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.01, 0.378), rot=(1.0, 0.0, 0.0, 0.0)), #pos=(0.0, -0.39, 0.6)
        
    )
   

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={{
            "goal": sim_utils.UsdFileCfg(
                usd_path=DEX_CUBE_USD_PATH,
                scale=(0.5, 0.5, 0.5),
            )
        }},
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=0.75, replicate_physics=True)
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -3.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 1000
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
"""

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # Write the generated content to the .py file
    with open(output_py_file, "w") as f:
        f.write(env_cfg_content)

    print(f"Environment configuration has been written to: {output_py_file}")

def strike_task_generate_env_cfg(urdf_path, mirror_urdf_path,observation_number, output_py_file):
    """
    Generate an environment configuration .py file based on the given URDF path and observation number.

    Args:
        urdf_path (str): Path to the URDF file.
        observation_number (int): Number of observations.
        output_py_file (str): Path to save the generated Python file.
    """

    # Helper function to parse URDF and extract joint and link names
    def parse_urdf(urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = []
        link_names = []

        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            if joint_name:
                joint_names.append(joint_name)

        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        return joint_names, link_names
    # Parse the URDF file
    left_joint_names, left_link_names = parse_urdf(urdf_path) #left hand

    #right hand =mirror hand
    right_joint_names, right_link_names = parse_urdf(mirror_urdf_path)


    # Create Python content for the environment configuration
    env_cfg_content = f"""
import os

from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sensors import ContactSensor,ContactSensorCfg
import numpy as np
import torch

@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={{
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        }},
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        }},
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }},
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={{
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )

from collections import defaultdict
@configclass
class EvolutionStrikeEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    # urdf_path = "{urdf_path}"
    # observation_number = {observation_number}

    # Actuated joints and fingertip links
    actuated_joint_names = {right_joint_names}
    finger_body_names = {right_link_names}

    #分离出指尖
    finger_links = defaultdict(list)
    # Step 1: 分组
    for name in finger_body_names:
        parts = name.split('_')
        if len(parts) != 3:
            continue
        finger_id = parts[1]
        joint_id = int(parts[2])
        # 跳过手掌 link_0_0
        if finger_id == '0':
            continue
        finger_links[finger_id].append((joint_id, name))

    # Step 2: 找最大 joint_id 的 link
    fingertip_body_names = []
    
    for finger_id, joints in finger_links.items():
        max_joint = max(joints, key=lambda x: x[0])
        fingertip_body_names.append(max_joint[1])  # only name

    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = len(actuated_joint_names)
    observation_space = len(actuated_joint_names)*3+len(fingertip_body_names)*13+16  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    
    # Robot configuration (using parsed joint names)
    
    #strike_hand 位置还得改 右手
    robot_cfg: ArticulationCfg = RIGHT_HAND_CFG.replace(prim_path="/World/envs/env_.*/RightRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.03, 0.275),
            rot=(0.0, 0.0, 1.0, 0),
            joint_pos={{".*": 0.0}},
            # joint_pos={{'link_0_0_to_link_1_0':1.0, 
            #            'link_1_0_to_link_1_1':1.5, 
            #            'link_1_1_to_link_1_2':1.5, 
            #            'link_0_0_to_link_2_0':0.0, 
            #            'link_2_0_to_link_2_1':0.0, 
            #            'link_2_1_to_link_2_2':0.0, 
            #            'link_2_2_to_link_2_3':0.0, 
            #            'link_0_0_to_link_3_0':0.0, 
            #            'link_3_0_to_link_3_1':0.0, 
            #            'link_3_1_to_link_3_2':0.0, 
            #            'link_3_2_to_link_3_3':0.0, 
            #            'link_0_0_to_link_4_0':0.0, 
            #            'link_4_0_to_link_4_1':0.0, 
            #            'link_4_1_to_link_4_2':0.0, 
            #            'link_4_2_to_link_4_3':0.0, 
            #            'link_0_0_to_link_5_0':0.0, 
            #            'link_5_0_to_link_5_1':0.0, 
            #            'link_5_1_to_link_5_2':0.0, 
            #            'link_5_2_to_link_5_3':0.0
            #            }},

        )
    )
    #cone_cfg
    Cone_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.01,
            height=0.10,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.005,  # 可以尝试增加此值
                # rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.01, 0.27),  #(0.055, -0.375, 0.42)
            rot=tuple(
                quat_from_angle_axis(
                    torch.tensor(-np.pi, dtype=torch.float32), #旋转180度
                    torch.tensor([0.0, 1.0, 0.0])) #绕y轴旋转 使尖端朝下
                    .tolist()),  
            ),#初始状态 
    )
    #strike_object_cfg
    strike_object_cfg:RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/strike_object",
        spawn=sim_utils.CuboidCfg(
            size=(0.2,0.2,0.20),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, #物体静止
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  # 可以尝试增加此值
                rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.05, 0.01, 0.10), rot=(1.0, 0.0, 0.0, 0.0)),#初始状态
    )
    
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/strike_object",
        filter_prim_paths_expr=["/World/envs/env_.*/Cone"],
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=0.75, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -1.0
    force_reward_scale=-10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 400
    fall_penalty = 0
    fall_dist = 0.045
    vel_obs_scale = 0.2
    success_tolerance = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
"""

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # Write the generated content to the .py file
    with open(output_py_file, "w") as f:
        f.write(env_cfg_content)

    print(f"Environment configuration has been written to: {output_py_file}")


def stonegrind_task_generate_env_cfg(urdf_path, mirror_urdf_path,observation_number, output_py_file):
    """
    Generate an environment configuration .py file based on the given URDF path and observation number.

    Args:
        urdf_path (str): Path to the URDF file.
        observation_number (int): Number of observations.
        output_py_file (str): Path to save the generated Python file.
    """

    # Helper function to parse URDF and extract joint and link names
    def parse_urdf(urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = []
        link_names = []

        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            if joint_name:
                joint_names.append(joint_name)

        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        return joint_names, link_names

    # Parse the URDF file
    left_joint_names, left_link_names = parse_urdf(urdf_path) #left hand

    #right hand =mirror hand
    right_joint_names, right_link_names = parse_urdf(mirror_urdf_path)


    # Create Python content for the environment configuration
    env_cfg_content = f"""
# from isaaclab_tasks.evolution_tasks.current_evolution_agent.current_hand_cfg import CURRENT_HAND_CFG #hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
import math

@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={{
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        }},
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        }},
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }},
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={{
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
from collections import defaultdict
@configclass
class EvolutionStoneGrindEnvCfg(DirectMARLEnvCfg):
    # Configuration for the environment

    # urdf_path = "{urdf_path}"
    # observation_number = {observation_number}

    # Actuated joints and fingertip links
    actuated_joint_names = {left_joint_names}
    finger_body_names = {left_link_names}

    #分离出指尖
    finger_links = defaultdict(list)
    # Step 1: 分组
    for name in finger_body_names:
        parts = name.split('_')
        if len(parts) != 3:
            continue
        finger_id = parts[1]
        joint_id = int(parts[2])
        # 跳过手掌 link_0_0
        if finger_id == '0':
            continue
        finger_links[finger_id].append((joint_id, name))

    # Step 2: 找最大 joint_id 的 link
    fingertip_body_names = []
    for finger_id, joints in finger_links.items():
        max_joint = max(joints, key=lambda x: x[0])
        fingertip_body_names.append(max_joint[1])  # only name


    # Environment settings
    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["right_hand", "left_hand"]
    action_spaces={{"right_hand": 0, "left_hand":  0}}
    # print(f"action_spaces1:{{action_spaces}}")
    action_spaces ={{"right_hand": len(actuated_joint_names), "left_hand":  len(actuated_joint_names)}}
    # print(f"action_spaces2:{{action_spaces}}")
    # action_space = len(actuated_joint_names)  # Action space based on number of actuated joints
    observation_spaces={{"right_hand": 0, "left_hand":  0}}
    observation_spaces = {{"right_hand": len(actuated_joint_names)*3+len(fingertip_body_names)*13+16, "left_hand":  len(actuated_joint_names)*3+len(fingertip_body_names)*13+13}}  # Observation space from input
    state_space = 2*(len(actuated_joint_names)*3+len(fingertip_body_names)*13)+16+13
    #asymmetric_obs = False
    #obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    
    # Robot configuration (using parsed joint names)
    #grasp_hand 位置还得改 左手
    left_robot_cfg: ArticulationCfg = LEFT_HAND_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(-0.707107, 0.707107, 0.0, 0),
            joint_pos={{".*": 0.0}},
        )
    )
    #strike_hand 位置还得改 右手
    right_robot_cfg: ArticulationCfg = RIGHT_HAND_CFG.replace(prim_path="/World/envs/env_.*/RightRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.03, 0.465),
            rot=(0.0, 0.0, 1.0, 0),
            joint_pos={{".*": 0.0}},
            # joint_pos={{'link_0_0_to_link_1_0':1.0, 
            #            'link_1_0_to_link_1_1':1.5, 
            #            'link_1_1_to_link_1_2':1.5, 
            #            'link_0_0_to_link_2_0':0.0, 
            #            'link_2_0_to_link_2_1':0.0, 
            #            'link_2_1_to_link_2_2':0.0, 
            #            'link_2_2_to_link_2_3':0.0, 
            #            'link_0_0_to_link_3_0':0.0, 
            #            'link_3_0_to_link_3_1':0.0, 
            #            'link_3_1_to_link_3_2':0.0, 
            #            'link_3_2_to_link_3_3':0.0, 
            #            'link_0_0_to_link_4_0':0.0, 
            #            'link_4_0_to_link_4_1':0.0, 
            #            'link_4_1_to_link_4_2':0.0, 
            #            'link_4_2_to_link_4_3':0.0, 
            #            'link_0_0_to_link_5_0':0.0, 
            #            'link_5_0_to_link_5_1':0.0, 
            #            'link_5_1_to_link_5_2':0.0, 
            #            'link_5_2_to_link_5_3':0.0
            #            }},

        )
    )
    #cone_cfg
    Cone_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.01,
            height=0.10,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.005,  # 可以尝试增加此值
                # rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.01, 0.45),  #(0.055, -0.375, 0.42)
            rot=tuple(
                quat_from_angle_axis(
                    torch.tensor(-np.pi, dtype=torch.float32), #旋转180度
                    torch.tensor([0.0, 1.0, 0.0])) #绕y轴旋转 使尖端朝下
                    .tolist()),  
            ),#初始状态 
    )
    #grasp_object_cfg
    grasp_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.005,  # 可以尝试增加此值
                # rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.01, 0.378),  
            rot=(1.0,0.0,0.0,0.0),#初始状态 
        )
    )
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        filter_prim_paths_expr=["/World/envs/env_.*/Cone"],
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    # fall_dist = 0.24
    transition_scale=0.5
    orientation_scale=0.5
    vel_obs_scale = 0.2
    act_moving_average = 1.0 #1.0 ???
    # reward-related scales
    # dist_reward_scale = 20.0

    # reward scales
    dist_reward_scale = 40.0#-1.0
    angle_reward_scale=-1.0
    force_reward_scale= -5.0
    action_penalty_scale = -0.0002
    reach_goal_bonus = 1000
    fall_penalty = 0
    grasp_dist = 0.025
    success_tolerance = 1
    max_consecutive_success = 0
    av_factor = 0.1
    fall_dist=0.06

    # # Reset parameters
    # reset_position_noise = 0.01  # Range of position at reset
    # reset_dof_pos_noise = 0.2  # Range of DOF position at reset
    # reset_dof_vel_noise = 0.0  # Range of DOF velocity at reset

    # # Reward scales
    # dist_reward_scale = -10.0
    # rot_reward_scale = 1.0
    # rot_eps = 0.1
    # action_penalty_scale = -0.0002
    # reach_goal_bonus = 250
    # fall_penalty = 0
    # fall_dist = 0.24
    # vel_obs_scale = 0.2
    # success_tolerance = 0.1
    # max_consecutive_success = 0
    # av_factor = 0.1
    # act_moving_average = 1.0
    # force_torque_obs_scale = 10.0
"""

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # Write the generated content to the .py file
    with open(output_py_file, "w") as f:
        f.write(env_cfg_content)

    print(f"Environment configuration has been written to: {output_py_file}")




def cube_task_generate_env_cfg(urdf_path, observation_number, output_py_file):
    """
    Generate an environment configuration .py file based on the given URDF path and observation number.

    Args:
        urdf_path (str): Path to the URDF file.
        observation_number (int): Number of observations.
        output_py_file (str): Path to save the generated Python file.
    """

    # Helper function to parse URDF and extract joint and link names
    def parse_urdf(urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joint_names = []
        link_names = []

        for joint in root.findall("joint"):
            joint_name = joint.get("name")
            if joint_name:
                joint_names.append(joint_name)

        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name:
                link_names.append(link_name)

        return joint_names, link_names

    # Parse the URDF file
    joint_names, link_names = parse_urdf(urdf_path)

    # Create Python content for the environment configuration
    env_cfg_content = f"""
from isaaclab_tasks.evolution_tasks.current_evolution_agent.current_hand_cfg import CURRENT_HAND_CFG
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
import math

@configclass
class EventCfg:
    # Configuration for randomization.
    # -- robot
    # 定义材料相关属性
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={{
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        }},
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )
    # 定义关节范围
    robot_joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",(-0.28, 0.26, 0.39)
        params={{
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        }},
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        }},
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={{
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }},
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={{
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        }},
    )

@configclass
class HandEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    urdf_path = "{urdf_path}"
    observation_number = {observation_number}

    # Actuated joints and fingertip links
    actuated_joint_names = {joint_names}
    fingertip_body_names = {link_names}

    # Environment settings
    decimation = 2
    episode_length_s = 10.0
    action_space = len(actuated_joint_names)  # Action space based on number of actuated joints
    observation_space = observation_number  # Observation space from input
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    #roll, pitch, yaw =  math.radians(180), math.radians(0), math.radians(0)
    roll, pitch, yaw =  math.radians(270), math.radians(0), math.radians(0)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    # Robot configuration (using parsed joint names)
    robot_cfg: ArticulationCfg = CURRENT_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(w, x, y, z),  # Quaternion rotation (w, x, y, z) of the root in simulation world frame. Defaults to (1.0, 0.0, 0.0, 0.0).
            joint_pos={{".*": 0.0}},
        )
    )

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{{ISAAC_NUCLEUS_DIR}}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.7, 0.7, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0), # 定义质量
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.02, 0.55), rot=(1.0, 0.0, 0.0, 0.0)), # 初始化位置
    )

    # goal object 加载目标对象
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={{
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{{ISAAC_NUCLEUS_DIR}}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        }},
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=0.75, replicate_physics=True)

    # Reset parameters
    reset_position_noise = 0.01  # Range of position at reset
    reset_dof_pos_noise = 0.2  # Range of DOF position at reset
    reset_dof_vel_noise = 0.0  # Range of DOF velocity at reset

    # Reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
"""

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_py_file), exist_ok=True)

    # Write the generated content to the .py file
    with open(output_py_file, "w") as f:
        f.write(env_cfg_content)

    print(f"Environment configuration has been written to: {output_py_file}")
