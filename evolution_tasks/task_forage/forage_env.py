from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate
from isaaclab_tasks.evolution_tasks.palm_coupling import apply_virtual_palm_coupling
from isaaclab_tasks.evolution_tasks.cartesian_hand_controller import MorphologyAwareFingertipIK, canonical_joint_observation

if TYPE_CHECKING:
    from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg


class ForageEnv(DirectRLEnv):
    """A hand uncovers a food item hidden under one or two movable leaf pieces."""

    cfg: ForageEnvCfg

    def __init__(self, cfg: ForageEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_hand_dofs = self.hand.num_joints
        self.num_joint_actions = 20
        self.actuated_dof_indices = [self.hand.joint_names.index(name) for name in self.cfg.actuated_joint_names]
        self.finger_bodies = [self.hand.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.joint_actions = torch.zeros((self.num_envs, self.num_joint_actions), device=self.device)
        self.wrist_actions = torch.zeros((self.num_envs, self.cfg.wrist_action_dim), device=self.device)
        self.wrist_offsets = torch.zeros_like(self.wrist_actions)
        self.wrist_limits = torch.tensor(self.cfg.wrist_translation_limits, device=self.device)
        # Retain the geometric evidence for the terminal reward.  DirectRLEnv
        # may reset an environment before a caller inspects its next state.
        self.success_leaf_distances = torch.zeros((self.num_envs, 2), device=self.device)
        # Emit the terminal task reward only once per episode.
        self.success_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        joint_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_limits[..., 0]
        self.hand_dof_upper_limits = joint_limits[..., 1]
        self.canonical_joint_names = tuple(self.cfg.actuated_joint_names)
        self.cartesian_ik = MorphologyAwareFingertipIK(
            self.hand, self.cfg.fingertip_body_names, num_envs=self.num_envs, device=self.device
        )

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
        if actions.shape[-1] != self.cfg.action_space:
            raise ValueError(f"Expected {self.cfg.action_space} forage actions, received {actions.shape[-1]}.")
        self.actions = actions.clone()
        self.joint_actions = actions[:, : self.num_joint_actions].clone()
        self.wrist_actions = torch.clamp(actions[:, self.num_joint_actions :], -1.0, 1.0)

    def _apply_action(self):
        targets = self.cartesian_ik.compute(self.joint_actions)
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        targets = saturate(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        self.cur_targets[:] = targets
        self.prev_targets[:] = targets
        self.hand.set_joint_position_target(targets)

        # Position-control the existing wrist/root frame, without changing the
        # hand topology.  The target remains relative to each environment's
        # initial root pose and is low-pass filtered to keep contacts stable.
        target_offset = self.wrist_actions * self.wrist_limits
        self.wrist_offsets = (
            self.cfg.wrist_moving_average * target_offset
            + (1.0 - self.cfg.wrist_moving_average) * self.wrist_offsets
        )
        root_state = self.hand.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins + self.wrist_offsets
        root_state[:, 7:] = 0.0
        self.hand.write_root_pose_to_sim(root_state[:, :7])
        self.hand.write_root_velocity_to_sim(root_state[:, 7:])

    def _get_observations(self):
        self._compute_intermediate_values()
        canonical_pos, canonical_vel = canonical_joint_observation(
            self.hand, self.canonical_joint_names, self.hand_dof_pos, self.hand_dof_vel,
            self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )
        obs = torch.cat((
            canonical_pos,
            canonical_vel,
            self.fingertip_pos.reshape(self.num_envs, -1),
            self.food_pose,
            self.leaf_one_pose,
            self.leaf_two_pose,
            self.cartesian_ik.morphology_descriptor(),
            self.actions,
        ), dim=-1)
        return {"policy": obs}

    def _get_rewards(self):
        self._compute_intermediate_values()
        leaf_distances = self._leaf_distances()
        success = torch.all(leaf_distances >= self.cfg.leaf_clear_distance, dim=-1)
        just_succeeded = success & ~self.success_achieved
        self.success_leaf_distances[just_succeeded] = leaf_distances[just_succeeded]
        self.success_achieved |= success
        return 1000.0 * just_succeeded.float()

    def _get_dones(self):
        self._compute_intermediate_values()
        uncovered = self._both_leaves_cleared()
        food_lost = torch.norm(self.food_pos[:, :2], dim=-1) > 0.22
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return food_lost | uncovered, time_out

    def _both_leaves_cleared(self) -> torch.Tensor:
        """Both leaf centres must clear the stage-specific horizontal distance."""
        return torch.all(self._leaf_distances() >= self.cfg.leaf_clear_distance, dim=-1)

    def _leaf_distances(self) -> torch.Tensor:
        return torch.stack((
            torch.norm(self.leaf_one_pos[:, :2] - self.food_pos[:, :2], dim=-1),
            torch.norm(self.leaf_two_pos[:, :2] - self.food_pos[:, :2], dim=-1),
        ), dim=-1)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)
        hand_state = self.hand.data.default_root_state[env_ids].clone()
        hand_state[:, :3] += self.scene.env_origins[env_ids]
        hand_state[:, 7:] = 0.0
        self.hand.write_root_state_to_sim(hand_state, env_ids)
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
        self.success_achieved[env_ids] = False
        self.wrist_offsets[env_ids] = 0.0
        self.cartesian_ik.reset(env_ids)
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
