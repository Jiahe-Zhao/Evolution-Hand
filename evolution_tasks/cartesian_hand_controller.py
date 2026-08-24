"""GPU-batched fingertip targets and morphology-aware differential IK."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate


NUM_FINGERS = 5
FINGERTIP_POSITION_ACTION_DIM = NUM_FINGERS * 3
FINGER_RESIDUAL_ACTION_DIM = NUM_FINGERS


def canonical_joint_observation(hand, canonical_joint_names, joint_pos, joint_vel, lower_limits, upper_limits):
    """Pad a variable-DOF morphology into a stable normalized joint interface."""
    num_envs = joint_pos.shape[0]
    device = joint_pos.device
    canonical_pos = torch.full((num_envs, len(canonical_joint_names)), -1.0, device=device)
    canonical_vel = torch.zeros_like(canonical_pos)
    active_action_indices = []
    active_dof_indices = []
    dof_count = joint_pos.shape[1]
    for action_index, joint_name in enumerate(canonical_joint_names):
        if joint_name not in hand.joint_names:
            continue
        dof_index = hand.joint_names.index(joint_name)
        if dof_index >= dof_count:
            continue
        active_action_indices.append(action_index)
        active_dof_indices.append(dof_index)
    if active_dof_indices:
        action_ids = torch.tensor(active_action_indices, dtype=torch.long, device=device)
        dof_ids = torch.tensor(active_dof_indices, dtype=torch.long, device=device)
        selected_lower = lower_limits[:, dof_ids]
        selected_upper = upper_limits[:, dof_ids]
        denominator = (selected_upper - selected_lower).clamp_min(1e-6)
        canonical_pos[:, action_ids] = 2.0 * (joint_pos[:, dof_ids] - selected_lower) / denominator - 1.0
        canonical_vel[:, action_ids] = joint_vel[:, dof_ids]
    return canonical_pos, canonical_vel


class MorphologyAwareFingertipIK:
    """Batched damped-least-squares IK for the currently loaded hand."""

    def __init__(
        self,
        hand,
        fingertip_body_names: Sequence[str],
        num_envs: int,
        device: str,
        *,
        position_scale=(0.025, 0.025, 0.025),
        residual_scale=0.018,
        damping=0.04,
        gain=0.8,
        reachability_tolerance=0.012,
    ):
        self.hand = hand
        self.num_envs = int(num_envs)
        self.device = device
        self.position_scale = torch.tensor(position_scale, dtype=torch.float32, device=self.device)
        self.residual_scale = float(residual_scale)
        self.damping = float(damping)
        self.gain = float(gain)
        self.reachability_tolerance = float(reachability_tolerance)
        self.body_ids: list[int] = []
        self.joint_ids: list[torch.Tensor] = []
        self.finger_names: list[str] = []

        for finger_id in range(1, NUM_FINGERS + 1):
            prefix = f"link_{finger_id}_"
            candidates = [name for name in fingertip_body_names if name.startswith(prefix) and name in hand.body_names]
            if not candidates:
                candidates = [
                    name for name in hand.body_names
                    if name.startswith(prefix) and name.rsplit("_", 1)[-1].isdigit()
                ]
            if not candidates:
                self.body_ids.append(-1)
                self.joint_ids.append(torch.empty(0, dtype=torch.long, device=self.device))
                self.finger_names.append("")
                continue
            fingertip = max(candidates, key=lambda name: int(name.rsplit("_", 1)[-1]))
            ids = [
                index for index, name in enumerate(hand.joint_names)
                if name.startswith(prefix) or f"to_link_{finger_id}_" in name
            ]
            self.body_ids.append(hand.body_names.index(fingertip))
            self.joint_ids.append(torch.tensor(ids, dtype=torch.long, device=self.device))
            self.finger_names.append(fingertip)

        self.active_fingers = [index for index, body_id in enumerate(self.body_ids) if body_id >= 0]
        if not self.active_fingers:
            raise ValueError("The loaded hand has no fingertip body available for Cartesian IK.")
        self.last_ik_error = torch.zeros((self.num_envs, NUM_FINGERS), device=self.device)
        self.last_ik_reachable = torch.zeros((self.num_envs, NUM_FINGERS), dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_ik_error.zero_()
            self.last_ik_reachable.zero_()
        else:
            self.last_ik_error[env_ids] = 0.0
            self.last_ik_reachable[env_ids] = False

    def _finger_jacobian_world(self, body_id: int, joint_ids: torch.Tensor) -> torch.Tensor:
        if body_id <= 0:
            raise ValueError(f"Invalid fingertip body index {body_id}; expected a non-root body.")
        jacobians = self.hand.root_physx_view.get_jacobians()
        return jacobians[:, body_id - 1, :3, joint_ids]

    def compute(self, actions: torch.Tensor, *, palm_center_local=(0.018, 0.0, 0.018)) -> torch.Tensor:
        expected_dim = FINGERTIP_POSITION_ACTION_DIM + FINGER_RESIDUAL_ACTION_DIM
        if actions.shape[-1] < expected_dim:
            raise ValueError(f"Cartesian hand action needs at least {expected_dim} values, got {actions.shape[-1]}.")

        num_envs = actions.shape[0]
        current_joint_pos = self.hand.data.joint_pos
        joint_targets = current_joint_pos.clone()
        root_quat = self.hand.data.root_quat_w
        current_tip_pos = self.hand.data.body_pos_w
        delta = actions[:, :FINGERTIP_POSITION_ACTION_DIM].reshape(num_envs, NUM_FINGERS, 3)
        delta = torch.clamp(delta, -1.0, 1.0) * self.position_scale
        residual = torch.clamp(actions[:, FINGERTIP_POSITION_ACTION_DIM:expected_dim], -1.0, 1.0)
        palm_local = torch.tensor(palm_center_local, dtype=torch.float32, device=self.device).expand(num_envs, -1)
        palm_world = self.hand.data.root_pos_w + quat_apply(root_quat, palm_local)

        self.last_ik_error.zero_()
        self.last_ik_reachable.zero_()
        for finger_id in self.active_fingers:
            body_id = self.body_ids[finger_id]
            joint_ids = self.joint_ids[finger_id]
            tip = current_tip_pos[:, body_id]
            toward_palm = palm_world - tip
            toward_palm = toward_palm / torch.linalg.vector_norm(toward_palm, dim=-1, keepdim=True).clamp_min(1e-6)
            desired_delta = quat_apply(root_quat, delta[:, finger_id])
            desired_delta = desired_delta + residual[:, finger_id: finger_id + 1] * self.residual_scale * toward_palm
            jacobian = self._finger_jacobian_world(body_id, joint_ids)
            jj_t = jacobian @ jacobian.transpose(1, 2)
            eye = torch.eye(3, dtype=jacobian.dtype, device=self.device).expand(num_envs, -1, -1)
            projected = torch.linalg.solve(jj_t + (self.damping ** 2) * eye, desired_delta.unsqueeze(-1)).squeeze(-1)
            delta_joint = self.gain * (jacobian.transpose(1, 2) @ projected.unsqueeze(-1)).squeeze(-1)
            joint_targets[:, joint_ids] = current_joint_pos[:, joint_ids] + delta_joint
            reconstruction_error = torch.linalg.vector_norm(
                jacobian @ delta_joint.unsqueeze(-1) - desired_delta.unsqueeze(-1), dim=1
            ).squeeze(-1)
            self.last_ik_error[:, finger_id] = reconstruction_error
            self.last_ik_reachable[:, finger_id] = torch.isfinite(reconstruction_error) & (reconstruction_error <= self.reachability_tolerance)

        limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        return torch.clamp(joint_targets, limits[..., 0], limits[..., 1])

    def morphology_descriptor(self) -> torch.Tensor:
        """Return local fingertip layout plus an active-finger mask."""
        root_pos = self.hand.data.root_pos_w
        root_quat = self.hand.data.root_quat_w
        descriptor = torch.zeros((self.num_envs, NUM_FINGERS, 3), device=self.device)
        for finger_id in self.active_fingers:
            descriptor[:, finger_id] = quat_apply(
                quat_conjugate(root_quat), self.hand.data.body_pos_w[:, self.body_ids[finger_id]] - root_pos
            )
        mask = torch.zeros((self.num_envs, NUM_FINGERS), device=self.device)
        mask[:, self.active_fingers] = 1.0
        return torch.cat((descriptor.reshape(self.num_envs, -1), mask), dim=-1)

    @property
    def reachable_rate(self) -> torch.Tensor:
        active = torch.tensor(self.active_fingers, dtype=torch.long, device=self.device)
        return self.last_ik_reachable[:, active].float().mean(dim=-1)
