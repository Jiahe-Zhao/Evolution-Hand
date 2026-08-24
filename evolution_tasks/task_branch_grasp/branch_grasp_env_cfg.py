import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg import CURRENT_HAND_CFG as RIGHT_HAND_CFG


EVOLUTION_ROOT = os.environ.get("EVOLUTION_ROOT", os.path.join(os.path.expanduser("~"), "Evolution_PC"))
_THUMB_JOINTS = ("link_0_0_to_link_1_0", "link_1_0_to_link_1_1", "link_1_1_to_link_1_2")
_FAST_LONG_FINGER_JOINTS = (
    "link_0_0_to_link_2_0", "link_2_0_to_link_2_1", "link_2_1_to_link_2_2", "link_2_2_to_link_2_3",
    "link_0_0_to_link_3_0", "link_3_0_to_link_3_1", "link_3_1_to_link_3_2", "link_3_2_to_link_3_3",
)
_SLOW_LONG_FINGER_JOINTS = (
    "link_0_0_to_link_4_0", "link_4_0_to_link_4_1", "link_4_1_to_link_4_2", "link_4_2_to_link_4_3",
    "link_0_0_to_link_5_0", "link_5_0_to_link_5_1", "link_5_1_to_link_5_2", "link_5_2_to_link_5_3",
)
_BRANCH_STIFFNESS = {name: 80.0 for name in _THUMB_JOINTS}
_BRANCH_STIFFNESS.update({name: 35.0 for name in _FAST_LONG_FINGER_JOINTS})
_BRANCH_STIFFNESS.update({name: 180.0 for name in _SLOW_LONG_FINGER_JOINTS})
_BRANCH_DAMPING = {name: 2.0 for name in _THUMB_JOINTS}
_BRANCH_DAMPING.update({name: 5.0 for name in _FAST_LONG_FINGER_JOINTS})
_BRANCH_DAMPING.update({name: 1.0 for name in _SLOW_LONG_FINGER_JOINTS})
# The source hand's implicit 1.0 stiffness leaves the 19 joints effectively
# static in this contact task.  Branch uses a local actuator override so its
# scripted and learned closing motion produces measurable joint motion.
# BranchGrasp must evaluate the same evolved morphology as the other tasks.
# Keep the task-specific reward and scene, but bind the robot to the current
# right-hand URDF generated for this individual.
BRANCH_HAND_CFG = RIGHT_HAND_CFG


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
    action_space = 20
    observation_space = 100
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    robot_cfg: ArticulationCfg = BRANCH_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
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
            # The branch-mounted contact sensor requires PhysX contact reporting.
            activate_contact_sensors=True,
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

    branch_contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/branch",
        # Keep contact channels separate so palm/root collisions cannot count as a grasp.
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/link_1_2",
            "/World/envs/env_.*/Robot/link_2_3",
            "/World/envs/env_.*/Robot/link_3_3",
            "/World/envs/env_.*/Robot/link_4_3",
            "/World/envs/env_.*/Robot/link_5_3",
        ],
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=True)

    # Stage one uses a reachable contact event. Stage two restores the full
    # force, hold duration, and pose-stability requirement.
    curriculum_stage = os.environ.get("EVOLUTION_CURRICULUM_STAGE", "stage2").lower()
    use_easy_curriculum = curriculum_stage == "stage1"
    branch_contact_force_threshold = 0.4 if use_easy_curriculum else 1.0
    # A branch wrap must use the thumb and at least two long fingers; a single
    # incidental fingertip contact is not a meaningful grasp.
    min_long_finger_contacts = 2
    branch_relative_position_tolerance = 0.012
    branch_relative_rotation_tolerance = 0.12
    require_pose_stability = not use_easy_curriculum
    branch_success_hold_steps = 5 if use_easy_curriculum else 15
    success_reward = 1000.0
    reset_dof_pos_noise = 0.05
    reset_dof_vel_noise = 0.0
    act_moving_average = 0.4
    # Maximum difference between any long finger's average normalized flexion
    # drive and the four-finger mean.  The thumb remains independent.
    finger_coordination_max_deviation = 0.18
    # Four long fingers advance by this shared target-angle speed; the thumb
    # remains independent and contact can stop an individual finger naturally.
    long_finger_joint_speed_rad_s = 1.2
