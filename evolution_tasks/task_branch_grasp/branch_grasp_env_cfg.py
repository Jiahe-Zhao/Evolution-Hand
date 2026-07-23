import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG as RIGHT_HAND_CFG


@configclass
class BranchGraspEnvCfg(DirectRLEnvCfg):
    actuated_joint_names = [
        "link_0_0_to_link_1_0",
        "link_1_0_to_link_1_1",
        "link_1_1_to_link_1_2",
        "link_0_0_to_link_2_0",
        "link_2_0_to_link_2_1",
        "link_2_1_to_link_2_2",
        "link_2_2_to_link_2_3",
        "link_0_0_to_link_3_0",
        "link_3_0_to_link_3_1",
        "link_3_1_to_link_3_2",
        "link_3_2_to_link_3_3",
        "link_0_0_to_link_4_0",
        "link_4_0_to_link_4_1",
        "link_4_1_to_link_4_2",
        "link_4_2_to_link_4_3",
        "link_0_0_to_link_5_0",
        "link_5_0_to_link_5_1",
        "link_5_1_to_link_5_2",
        "link_5_2_to_link_5_3",
    ]
    fingertip_body_names = [
        "link_1_2",
        "link_2_3",
        "link_3_3",
        "link_4_3",
        "link_5_3",
    ]

    decimation = 2
    episode_length_s = 5.0
    action_space = len(actuated_joint_names)
    observation_space = len(actuated_joint_names) * 3 + len(fingertip_body_names) * 3 + 7
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    robot_cfg: ArticulationCfg = RIGHT_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.36),
            rot=(0.5, 0.5, 0.5, 0.5),
            joint_pos={".*": 0.0},
        )
    )

    branch_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/branch",
        spawn=sim_utils.CylinderCfg(
            radius=0.012,
            height=0.18,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.29, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.30),
            rot=(0.707107, 0.0, 0.707107, 0.0),
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=True)

    action_penalty_scale = 1.0e-4
    fingertip_tracking_reward_scale = 2.0
    branch_target_offset = (0.0, 0.0, 0.02)
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.0
    act_moving_average = 0.4
