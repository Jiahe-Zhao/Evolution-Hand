# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor,ContactSensorCfg


LOCAL_SHADOW_HAND_USD = os.environ.get("LOCAL_SHADOW_HAND_USD")
SHADOW_HAND_TASK_CFG = SHADOW_HAND_CFG.replace(
    spawn=SHADOW_HAND_CFG.spawn.replace(
        usd_path=LOCAL_SHADOW_HAND_USD or SHADOW_HAND_CFG.spawn.usd_path,
    )
)


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
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
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
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


@configclass
class StoneGrindShadowHandEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.5
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {"right_hand": 20, "left_hand": 20}
    observation_spaces = {"right_hand": 149, "left_hand": 146}
    state_space = 297 #307
    transition_scale=0.5
    orientation_scale=0.5
    # simulation
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
    # robot
    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]
    #grasp_hand 位置还得改 左手
    left_robot_cfg: ArticulationCfg = SHADOW_HAND_TASK_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        # fix_base=False,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(0.382683, 0.0, 0.0, -0.92388),
            joint_pos={".*": 0.0},
        )
    )
    #strike_hand 位置还得改 右手
    right_robot_cfg: ArticulationCfg = SHADOW_HAND_TASK_CFG.replace(prim_path="/World/envs/env_.*/RightRobot").replace(
        # fix_base=False,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.55, 0.52),
                rot=(0.653281, 0.270598,0.653281,  -0.270598),
                # rot=(1.0, 0.0,0.0,  0.0),
                joint_pos={
                    # 设置每个关节的初始角度，以模拟抓取锥体的姿态 
                    "robot0_WRJ1": 0.0, # 手腕 上下[-0.489, 0.140]
                    "robot0_WRJ0": 0.0, #手腕 左右[-0.698, 0.489]
                    #食指
                    "robot0_FFJ1": 0.2,  # 食指外关节弯曲
                    "robot0_FFJ2": 0.3,  # 食指根部弯曲
                    "robot0_FFJ3": 0.3,  # 食指根部上下 [-0.349, 0.349]
                    #中指
                    "robot0_MFJ1": 0.2,  # 中指外关节弯曲
                    "robot0_MFJ2": 0.3,  # 中指根部弯曲
                    "robot0_MFJ3": 0.3,  # 中指根部上下 [-0.349, 0.349]
                    #无名指
                    "robot0_RFJ1": 0.2,  # 无名指外关节弯曲
                    "robot0_RFJ2": 0.3,  # 无名指根部弯曲
                    "robot0_RFJ3": 0.3,  # 无名指根部上下[-0.349, 0.349]
                    #小指
                    "robot0_LFJ1": 0.3,  # 小指外关节弯曲
                    "robot0_LFJ2": 0.4,  # 小指根部弯曲
                    "robot0_LFJ3": 0.3,  # 小指根部上下 [-0.349, 0.349]
                    "robot0_LFJ4": 0.2,  
                    #拇指
                    "robot0_THJ0": -1.5,    # 拇指弯曲[-1.571, 0.000] -0.5
                    "robot0_THJ1": 0,  #  拇指关节旋转[-0.524, 0.524] 0.5
                    "robot0_THJ2": 0.2,  #  拇指关节旋转[-0.209, 0.209] -0.2
                    "robot0_THJ3": 1.2,  #  拇指根部旋转[[0.000, 1.222]
                    "robot0_THJ4": 0.8   # 拇指根部弯曲
                },
        )
    )
    
    #cone_cfg
    Cone_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.02,
            height=0.20,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=500.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  # 可以尝试增加此值
                rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.25, 0.25, 0.55),  #(0.055, -0.375, 0.42)
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
            radius=0.04,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
            # rot=(0.653281, 0.270598,0.653281,  -0.270598),
            # # rot=(1.0, 0.0,0.0,  0.0),
            # joint_pos={{".*": 0.0}},#初始位置需要修改
                max_depenetration_velocity=500.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  # 可以尝试增加此值
                rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.28, 0.26, 0.39),  
            rot=(1.0,0.0,0.0,0.0),#初始状态 
        )
    )
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        filter_prim_paths_expr=["/World/envs/env_.*/Cone"],
    )
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.0335,
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
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.0335,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=20, env_spacing=1.5, replicate_physics=True)

    # reset
    reset_position_noise = 0.001  # range of position at reset
    reset_dof_pos_noise = 0.02  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    # fall_dist = 0.24
    vel_obs_scale = 0.2
    act_moving_average = 1.0 #1.0 ???
    # reward-related scales
    # dist_reward_scale = 20.0

    # reward scales
    dist_reward_scale = -1.20
    angle_reward_scale=-4.5
    force_reward_scale= -5.0
    action_penalty_scale = -0.002
    reach_goal_bonus = 400
    fall_penalty = 0
    fall_dist = 0.05
    success_tolerance = 1
    max_consecutive_success = 0
    av_factor = 0.1
