from isaaclab.utils import configclass

from isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg import EvolutionGraspEnvCfg


@configclass
class CarryEnvCfg(EvolutionGraspEnvCfg):
    """Grasp scene with the food sphere already supported in the palm at reset."""

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0

    def __post_init__(self):
        super().__post_init__()
        # Unlike grasp's vertical approach pose, carry begins with an open,
        # horizontal palm so gravity seats the sphere against the hand.
        self.robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        # The model root is at the wrist. Shift along the finger direction to
        # center the sphere in the open palm rather than beside the wrist.
        self.grasp_object_cfg.init_state.pos = (-0.060, 0.0, 0.390)
        self.grasp_object_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
