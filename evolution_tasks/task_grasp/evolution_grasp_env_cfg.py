
from isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg import CURRENT_HAND_CFG as LEFT_HAND_CFG#hand cfg需要修改

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG  as RIGHT_HAND_CFG#hand cfg需要修改

import os

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
        params={
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0), # 禁用接触回弹，避免指尖与球持续弹跳
            "num_buckets": 250,
        },
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # 定义关节范围
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    # tendon
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    # 对象的材料属性
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    # 操作对象的质量分布
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    # 重力分布
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

from collections import defaultdict
@configclass
class EvolutionGraspEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    # urdf_path = "/share/home/zjh/Evolution/Isaaclab_other/agent_for_isaaclab/urdf/current_agent.urdf"
    

    # Actuated joints and fingertip links
    actuated_joint_names = ['link_0_0_to_link_1_0', 'link_1_0_to_link_1_1', 'link_1_1_to_link_1_2', 'link_0_0_to_link_2_0', 'link_2_0_to_link_2_1', 'link_2_1_to_link_2_2', 'link_2_2_to_link_2_3', 'link_0_0_to_link_3_0', 'link_3_0_to_link_3_1', 'link_3_1_to_link_3_2', 'link_3_2_to_link_3_3', 'link_0_0_to_link_4_0', 'link_4_0_to_link_4_1', 'link_4_1_to_link_4_2', 'link_4_2_to_link_4_3', 'link_0_0_to_link_5_0', 'link_5_0_to_link_5_1', 'link_5_1_to_link_5_2', 'link_5_2_to_link_5_3']
    finger_body_names = ['link_0_0', 'link_1_0', 'link_1_1', 'link_1_2', 'link_2_0', 'link_2_1', 'link_2_2', 'link_2_3', 'link_3_0', 'link_3_1', 'link_3_2', 'link_3_3', 'link_4_0', 'link_4_1', 'link_4_2', 'link_4_3', 'link_5_0', 'link_5_1', 'link_5_2', 'link_5_3']

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
    
    # 15 fingertip displacement targets + 5 task-level closure residuals.
    action_space = 20
    # Full observation plus the morphology descriptor (15 positions + 5 mask).
    observation_space = 159
   
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
            restitution=0.0,
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
            # Start from a finger-pad pre-grasp instead of an open hand.
            joint_pos={".*": 0.35},
        )
    )
    
    #grasp_object_cfg
    grasp_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
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
    # Runtime reset places the ball at proximal_finger_region_center in the
    # hand frame. This placeholder only defines the asset's default state.
            pos=(0.010, 0.005, 0.365),
            rot=(1.0,0.0,0.0,0.0),#初始状态 
        )
    )
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        # One channel per fingertip, ordered thumb then four long fingers.
        filter_prim_paths_expr=[
            "/World/envs/env_.*/LeftRobot/link_1_2",
            "/World/envs/env_.*/LeftRobot/link_2_3",
            "/World/envs/env_.*/LeftRobot/link_3_3",
            "/World/envs/env_.*/LeftRobot/link_4_3",
            "/World/envs/env_.*/LeftRobot/link_5_3",
        ],
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=20, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.0
    curriculum_stage = os.environ.get("EVOLUTION_CURRICULUM_STAGE", "stage2").lower()
    # First establish stable distal-finger support from a reproducible pose;
    # then reintroduce a modest joint perturbation for the strict stage.
    reset_dof_pos_noise = 0.0 if curriculum_stage == "stage1" else 0.05
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
    dist_reward_scale = 0.0
    angle_reward_scale=-3.0
    force_reward_scale= 0.0
    action_penalty_scale = 0.0
    reach_goal_bonus = 1000
    fall_penalty = 0
    # This box covers the visible central palm between the five fingers.  It
    # intentionally excludes the root-side edge that previously permitted
    # visual "edge support" to count as a grasp.
    visual_palm_region_center = (0.018, 0.0, 0.018)
    visual_palm_region_half_extents = (0.018, 0.024, 0.018)
    # The middle three first phalanges form a stable shelf just distal to the
    # palm. It is added to, rather than replacing, the original palm region.
    proximal_finger_region_center = (-0.025, -0.025, 0.030)
    proximal_finger_region_half_extents = (0.020, 0.020, 0.015)
    # Dynamic distal-finger enclosure: require the ball to be close to at least
    # two terminal phalanges, so a widely open hand cannot obtain a false success.
    distal_region_margin = 0.024
    distal_contact_radius = 0.060
    min_distal_nearby_fingers = 2
    spawn_on_distal_fingers = True
    distal_support_body_names = ("link_2_3", "link_3_3", "link_4_3")
    # World-frame upward offset: sphere radius plus a small contact clearance.
    distal_support_offset = (0.0, 0.0, 0.032)
    palm_region_reward_scale = 0.0
    # Three sparse milestones are active in both training stages and evaluation.
    # Stage1/Stage2 only differ in training budget and reset perturbation.
    m1_contact_force_threshold = 0.10  # any fingertip
    m2_contact_force_threshold = 0.10  # thumb plus another fingertip
    m3_contact_force_threshold = 0.25  # all five fingertips
    m1_hold_steps = 3
    m2_hold_steps = 5
    m3_hold_steps = 10
    m1_reward = 50.0
    m2_reward = 250.0
    m3_reward = 1000.0
    # Retained for compatibility with old reports; success is always M3 now.
    require_full_hand_contact = True
    full_hand_contact_force_threshold = m3_contact_force_threshold
    thumb_contact_index = 0
    required_fingertip_count = 5
    min_success_hold_steps = m3_hold_steps
    # grasp_dist = 0.025
    success_tolerance = 3
    max_consecutive_success = 0
    av_factor = 0.1
    fall_dist=0.15
