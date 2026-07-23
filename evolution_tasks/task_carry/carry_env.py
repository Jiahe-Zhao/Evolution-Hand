import torch

from isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env import EvolutionGraspEnv


class CarryEnv(EvolutionGraspEnv):
    """Carry-start variant of the grasp environment with an in-palm object reset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Grasp applies a 10 N test load at construction. Carry starts from a
        # naturally supported object, so only gravity should act initially.
        zero_forces = torch.zeros(
            (self.num_envs, self.grasp_object.num_bodies, 3), device=self.device
        )
        self.grasp_object.set_external_force_and_torque(forces=zero_forces, torques=zero_forces)
        self.grasp_object.write_data_to_sim()

        self._palm_body_ids = [
            self.hand.body_names.index(name)
            for name in ("link_1_0", "link_2_0", "link_3_0", "link_4_0", "link_5_0")
        ]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        # Build the palm center from the actual finger-root links, not the
        # wrist-frame origin used by the source grasp task.
        palm_center = self.hand.data.body_pos_w[env_ids][:, self._palm_body_ids].mean(dim=1)
        state = self.grasp_object.data.default_root_state[env_ids].clone()
        state[:, :3] = palm_center
        state[:, 2] += 0.032
        state[:, 7:] = 0.0
        self.grasp_object.write_root_state_to_sim(state, env_ids)
