from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_mul, sample_uniform, saturate

if TYPE_CHECKING:
    from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg


class BranchGraspEnv(DirectRLEnv):
    cfg: BranchGraspEnvCfg

    def __init__(self, cfg: BranchGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints
        self.actuated_dof_indices = [self.hand.joint_names.index(name) for name in self.cfg.actuated_joint_names]
        self.finger_bodies = [self.hand.body_names.index(name) for name in self.cfg.fingertip_body_names]
        self.num_fingertips = len(self.finger_bodies)

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float32, device=self.device)
        self.prev_targets = torch.zeros_like(self.hand_dof_targets)
        self.cur_targets = torch.zeros_like(self.hand_dof_targets)
        self.actions = torch.zeros(
            (self.num_envs, len(self.cfg.actuated_joint_names)), dtype=torch.float32, device=self.device
        )

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        self.branch_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_branch_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_branch_relative_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.branch = RigidObject(self.cfg.branch_cfg)
        self.branch_contact_sensor = ContactSensor(self.cfg.branch_contact_sensor_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["branch"] = self.branch
        self.scene.sensors["branch_contact_sensor"] = self.branch_contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
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
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.hand_dof_vel,
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.branch_pose,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        fingertip_forces = torch.norm(self.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1)
        thumb_contact = fingertip_forces[:, 0] >= self.cfg.branch_contact_force_threshold
        other_finger_contact = torch.any(
            fingertip_forces[:, 1:] >= self.cfg.branch_contact_force_threshold, dim=-1
        )
        relative_pos, relative_quat = self._branch_relative_pose()
        position_delta = torch.norm(relative_pos - self.previous_branch_relative_pos, dim=-1)
        quat_dot = torch.sum(relative_quat * self.previous_branch_relative_quat, dim=-1).abs().clamp(max=1.0)
        rotation_delta = 2.0 * torch.acos(quat_dot)
        pose_stable = (position_delta <= self.cfg.branch_relative_position_tolerance) & (
            rotation_delta <= self.cfg.branch_relative_rotation_tolerance
        )
        if not self.cfg.require_pose_stability:
            pose_stable = torch.ones_like(pose_stable)
        qualified = thumb_contact & other_finger_contact & pose_stable
        self.branch_success_streak = torch.where(qualified, self.branch_success_streak + 1, 0)
        self.previous_branch_relative_pos = relative_pos
        self.previous_branch_relative_quat = relative_quat
        just_succeeded = self.branch_success_streak == self.cfg.branch_success_hold_steps

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["branch_thumb_force"] = fingertip_forces[:, 0].mean()
        self.extras["log"]["branch_other_finger_force"] = fingertip_forces[:, 1:].amax(dim=-1).mean()
        self.extras["log"]["branch_qualified_rate"] = qualified.float().mean()
        self.extras["log"]["branch_hold_steps"] = self.branch_success_streak.float().mean()
        return self.cfg.success_reward * just_succeeded.float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.branch_success_streak >= self.cfg.branch_success_hold_steps
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)

        branch_state = self.branch.data.default_root_state[env_ids].clone()
        branch_state[:, 0:3] = branch_state[:, 0:3] + self.scene.env_origins[env_ids]
        branch_state[:, 7:] = 0.0
        self.branch.write_root_state_to_sim(branch_state, env_ids)

        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta
        dof_vel = torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device)

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.branch_success_streak[env_ids] = 0

        self._compute_intermediate_values()
        relative_pos, relative_quat = self._branch_relative_pose()
        self.previous_branch_relative_pos[env_ids] = relative_pos[env_ids]
        self.previous_branch_relative_quat[env_ids] = relative_quat[env_ids]

    def _compute_intermediate_values(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.unsqueeze(1)
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.branch_pos = self.branch.data.root_pos_w - self.scene.env_origins
        self.branch_rot = self.branch.data.root_quat_w
        self.branch_pose = torch.cat((self.branch_pos, self.branch_rot), dim=-1)

    def _branch_relative_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        relative_pos = self.branch.data.root_pos_w - self.hand.data.root_pos_w
        relative_quat = quat_mul(quat_conjugate(self.hand.data.root_quat_w), self.branch.data.root_quat_w)
        return relative_pos, relative_quat


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / (upper - lower) - 1.0
