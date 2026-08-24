import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG as RIGHT_HAND_CFG


@configclass
class ForageEnvCfg(DirectRLEnvCfg):
    """Minimal surface-foraging task: uncover a food item by moving leaf litter."""

    actuated_joint_names = [
        "link_0_0_to_link_1_0", "link_1_0_to_link_1_1", "link_1_1_to_link_1_2",
        "link_0_0_to_link_2_0", "link_2_0_to_link_2_1", "link_2_1_to_link_2_2", "link_2_2_to_link_2_3",
        "link_0_0_to_link_3_0", "link_3_0_to_link_3_1", "link_3_1_to_link_3_2", "link_3_2_to_link_3_3",
        "link_0_0_to_link_4_0", "link_4_0_to_link_4_1", "link_4_1_to_link_4_2", "link_4_2_to_link_4_3",
        "link_0_0_to_link_5_0", "link_5_0_to_link_5_1", "link_5_1_to_link_5_2", "link_5_2_to_link_5_3",
    ]
    fingertip_body_names = ["link_1_2", "link_2_3", "link_3_3", "link_4_3", "link_5_3"]

    decimation = 2
    episode_length_s = 6.0
    # Nineteen finger joints plus Cartesian translation of the wrist base.
    # The hand URDF is unchanged: the final three action channels control the
    # relative x/y/z position of its existing root frame.
    wrist_action_dim = 3
    action_space = 23
    # Joint state, fingertip positions, food pose, two leaf poses, and action.
    observation_space = 117
    state_space = 0

    # Both stages use the final two-leaf scene and one sparse terminal reward.
    # Stage one only reduces the clearance requirement to establish a reliable
    # surface-sweeping behavior before the strict 7.5 cm criterion.
    curriculum_stage = os.environ.get("EVOLUTION_FORAGE_CURRICULUM_STAGE", "stage2").lower()
    active_leaf_count = 2
    leaf_clear_distance = 0.040 if curriculum_stage == "stage1" else 0.075

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.75),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # Match the Branch task: the palm and fingers face the work surface.
    robot_cfg: ArticulationCfg = RIGHT_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.21),
            rot=(0.5, 0.5, 0.5, 0.5),
            joint_pos={".*": 0.0},
        )
    )

    food_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/food",
        # A fixed soil/food patch gives the leaves a planar resting surface.
        # Unlike the former sphere, it cannot roll or kick leaves away at reset.
        spawn=sim_utils.CuboidCfg(
            size=(0.140, 0.120, 0.100),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.68, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.85, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.050)),
    )

    # Two thin movable leaves overlap above the food. They deliberately use simple convex boxes
    # so the first task remains stable and highly parallelizable.
    leaf_one_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/leaf_one",
        spawn=sim_utils.CuboidCfg(
            size=(0.100, 0.065, 0.004),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.50, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.012, 0.0, 0.1022)),
    )
    leaf_two_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/leaf_two",
        spawn=sim_utils.CuboidCfg(
            size=(0.095, 0.060, 0.004),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.31, 0.58, 0.16)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.50, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.012, 0.002, 0.1064),
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=1.0, replicate_physics=True)
    action_penalty_scale = 1.0e-4
    uncover_reward_scale = 2.0
    success_reward = 5.0
    # A leaf is clear only once its centre has moved beyond the rectangular
    # support boundary, not merely across a radial distance threshold.
    support_half_extent_x = 0.070
    support_half_extent_y = 0.060
    support_clearance_margin = 0.010
    reset_dof_pos_noise = 0.02
    act_moving_average = 0.4
    wrist_translation_limits = (0.08, 0.10, 0.06)
    wrist_moving_average = 0.30
