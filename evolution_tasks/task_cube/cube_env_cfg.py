# from isaaclab_tasks.evolution_tasks.current_evolution_agent.human_hand_414 import CURRENT_HAND_CFG
from isaaclab_tasks.evolution_tasks.current_evolution_agent.human_mirror_hand_414 import CURRENT_HAND_CFG
# from isaaclab_tasks.evolution_tasks.current_evolution_agent.current_hand_cfg import CURRENT_HAND_CFG
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
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3), # 静摩擦范围
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0), # 回弹
            "num_buckets": 250,
        },
    )
    # 定义关节阻尼与刚度
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # 定义关节范围
    robot_joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
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
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
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
            "restitution_range": (1.0, 1.0),
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

@configclass
class HandEnvCfg(DirectRLEnvCfg):
    # Configuration for the environment

    urdf_path = "/share/home/zjh/Evolution/Isaaclab_other/human_mirror_hand_414.urdf"

    # urdf_path = "/share/home/zjh/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/current_evolution_agent/urdf/current_agent.urdf"
    observation_number = 341

    # Actuated joints and fingertip links
    actuated_joint_names = ['link_0_0_to_link_1_0', 'link_1_0_to_link_1_1', 'link_1_1_to_link_1_2', 'link_0_0_to_link_2_0', 'link_2_0_to_link_2_1', 'link_2_1_to_link_2_2', 'link_2_2_to_link_2_3', 'link_0_0_to_link_3_0', 'link_3_0_to_link_3_1', 'link_3_1_to_link_3_2', 'link_3_2_to_link_3_3', 'link_0_0_to_link_4_0', 'link_4_0_to_link_4_1', 'link_4_1_to_link_4_2', 'link_4_2_to_link_4_3', 'link_0_0_to_link_5_0', 'link_5_0_to_link_5_1', 'link_5_1_to_link_5_2', 'link_5_2_to_link_5_3']
    fingertip_body_names = ['link_0_0', 'link_1_0', 'link_1_1', 'link_1_2', 'link_2_0', 'link_2_1', 'link_2_2', 'link_2_3', 'link_3_0', 'link_3_1', 'link_3_2', 'link_3_3', 'link_4_0', 'link_4_1', 'link_4_2', 'link_4_3', 'link_5_0', 'link_5_1', 'link_5_2', 'link_5_3']

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
            joint_pos={".*": 0.0},
        )
    )

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
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
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
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
