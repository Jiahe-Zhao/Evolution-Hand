from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate

if TYPE_CHECKING:
    from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg


class ForageEnv(DirectRLEnv):
    """A hand uncovers a food item hidden under one or two movable leaf pieces."""

    cfg: ForageEnvCfg

    def __init__(self, cfg: ForageEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_hand_dofs = self.hand.num_joints
        self.actuated_dof_indices = [self.hand.joint_names.index(name) for name in self.cfg.actuated_joint_names]
        self.finger_bodies = [self.hand.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        joint_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_limits[..., 0]
        self.hand_dof_upper_limits = joint_limits[..., 1]

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.food = RigidObject(self.cfg.food_cfg)
        self.leaf_one = RigidObject(self.cfg.leaf_one_cfg)
        self.leaf_two = RigidObject(self.cfg.leaf_two_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["food"] = self.food
        self.scene.rigid_objects["leaf_one"] = self.leaf_one
        self.scene.rigid_objects["leaf_two"] = self.leaf_two
        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.hand.set_joint_position_target(self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices)

    def _get_observations(self):
        self._compute_intermediate_values()
        obs = torch.cat((
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.hand_dof_vel,
            self.fingertip_pos.reshape(self.num_envs, -1),
            self.food_pose,
            self.leaf_one_pose,
            self.leaf_two_pose,
            self.actions,
        ), dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        self._compute_intermediate_values()
        leaf_distances = torch.stack((
            torch.norm(self.leaf_one_pos[:, :2] - self.food_pos[:, :2], dim=-1),
            torch.norm(self.leaf_two_pos[:, :2] - self.food_pos[:, :2], dim=-1),
        ), dim=-1)
        # Both pieces must be moved laterally away from the food for a robust reveal.
        uncovered = leaf_distances.min(dim=-1).values
        action_penalty = torch.sum(self.actions.square(), dim=-1)
        reward = self.cfg.uncover_reward_scale * torch.clamp(uncovered / self.cfg.reveal_distance, max=1.0)
        reward -= self.cfg.action_penalty_scale * action_penalty
        reward += self.cfg.success_reward * (uncovered > self.cfg.reveal_distance)
        return reward

    def _get_dones(self):
        self._compute_intermediate_values()
        food_lost = torch.norm(self.food_pos[:, :2], dim=-1) > 0.22
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return food_lost, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)
        for asset in (self.food, self.leaf_one, self.leaf_two):
            state = asset.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            state[:, 7:] = 0.0
            asset.write_root_state_to_sim(state, env_ids)
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * (delta_min + (delta_max - delta_min) * 0.5 * noise)
        dof_vel = torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device)
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _compute_intermediate_values(self):
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies] - self.scene.env_origins.unsqueeze(1)
        self.food_pos = self.food.data.root_pos_w - self.scene.env_origins
        self.leaf_one_pos = self.leaf_one.data.root_pos_w - self.scene.env_origins
        self.leaf_two_pos = self.leaf_two.data.root_pos_w - self.scene.env_origins
        self.food_pose = torch.cat((self.food_pos, self.food.data.root_quat_w), dim=-1)
        self.leaf_one_pose = torch.cat((self.leaf_one_pos, self.leaf_one.data.root_quat_w), dim=-1)
        self.leaf_two_pose = torch.cat((self.leaf_two_pos, self.leaf_two.data.root_quat_w), dim=-1)


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / (upper - lower) - 1.0
