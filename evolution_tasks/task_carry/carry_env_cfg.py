from isaaclab.utils import configclass

from isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg import EvolutionGraspEnvCfg


@configclass
class CarryEnvCfg(EvolutionGraspEnvCfg):
    """Grasp scene with the food sphere already supported in the palm at reset."""

    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0

    def __post_init__(self):
        super().__post_init__()
        # The grasp task's support reference is 4 cm below its spawn point.
        # Spawn here directly so carry starts with the ball on the hand.
        self.grasp_object_cfg.init_state.pos = (-0.05, 0.01, 0.338)
        self.grasp_object_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
