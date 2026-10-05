from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET
from pathlib import Path
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, sample_uniform, saturate
from isaaclab_tasks.evolution_tasks.palm_coupling import (
    apply_branch_finger_coordination,
    finger_flexion_scores,
)
from isaaclab_tasks.evolution_tasks.cartesian_hand_controller import MorphologyAwareFingertipIK, canonical_joint_observation, resolve_fingertip_body_names

if TYPE_CHECKING:
    from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg


class BranchGraspEnv(DirectRLEnv):
    cfg: BranchGraspEnvCfg

    def __init__(self, cfg: BranchGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints
        # Preserve the canonical policy interface while allowing evolution to
        # remove distal links and therefore their associated joints.
        self.canonical_joint_names = tuple(self.cfg.actuated_joint_names)
        self.actuated_dof_indices = [
            self.hand.joint_names.index(name)
            for name in self.canonical_joint_names
            if name in self.hand.joint_names
        ]
        self.active_fingertip_names = resolve_fingertip_body_names(self.hand, self.cfg.fingertip_body_names)
        self.finger_bodies = [self.hand.body_names.index(name) for name in self.active_fingertip_names]
        self.thumb_tip_body_id = self.finger_bodies[0]
        self.long_finger_tip_body_ids = self.finger_bodies[1:]
        self.num_fingertips = len(self.finger_bodies)

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float32, device=self.device)
        self.prev_targets = torch.zeros_like(self.hand_dof_targets)
        self.cur_targets = torch.zeros_like(self.hand_dof_targets)
        self.actions = torch.zeros(
            (self.num_envs, 20), dtype=torch.float32, device=self.device
        )
        self.finger_action_scores = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        self.long_finger_velocity_spread = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.long_finger_joint_groups = [
            [
                self.hand.joint_names.index(name)
                for name in self.canonical_joint_names
                if (name.startswith(f"link_{finger_id}_") or f"to_link_{finger_id}_" in name)
                and not name.endswith("_mcp_spread_joint")
                and name in self.hand.joint_names
            ]
            for finger_id in range(2, 6)
        ]
        self.long_finger_joint_ids = [joint_id for group in self.long_finger_joint_groups for joint_id in group]
        self.previous_long_finger_joint_scores = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.long_finger_joint_scores = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.long_finger_joint_velocity_scores = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.long_finger_joint_velocity_spread = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]
        self.proximal_branch_body_ids = [
            self.hand.body_names.index(name)
            for name in self.cfg.branch_support_body_names
            if name in self.hand.body_names
        ]
        if not self.proximal_branch_body_ids:
            raise RuntimeError("BranchGrasp requires at least one configured proximal support body.")
        self._branch_collision_capsules = self._load_collision_capsules()
        self.cartesian_ik = MorphologyAwareFingertipIK(
            self.hand, self.cfg.fingertip_body_names, num_envs=self.num_envs, device=self.device
        )
        self.scripted_joint_target = None

        self.branch_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_branch_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_branch_relative_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        # PhysX body_names is unavailable until the scene is initialized.  Keep
        # the canonical sensor paths here; the IK/controller handles missing
        # links after articulation initialization.
        available_tips = list(self.cfg.fingertip_body_names)
        # Keep the reward sensor fingertip-only. A separate all-body sensor
        # audits initial placement without changing the reward channel order.
        self.cfg.branch_contact_sensor_cfg.filter_prim_paths_expr = [
            f"/World/envs/env_.*/Robot/{name}" for name in available_tips
        ]
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
        self.finger_action_scores = self.actions[:, 15:20]

    def _apply_action(self):
        if os.environ.get("EVOLUTION_BRANCH_BC_MODE") == "1":
            # Four policy channels per finger command its available joints.
            # Long-finger side spread remains neutral as in the task dynamics.
            targets = self.hand.data.joint_pos.clone()
            lower, upper = self.hand_dof_lower_limits, self.hand_dof_upper_limits
            for finger in range(5):
                joint_ids = self.cartesian_ik.joint_ids[finger].tolist()
                if finger > 0:
                    joint_ids = [index for index in joint_ids if "spread" not in self.hand.joint_names[index]]
                for channel, joint_id in enumerate(joint_ids[:4]):
                    command = self.actions[:, 4 * finger + channel].clamp(-1.0, 1.0)
                    targets[:, joint_id] = lower[:, joint_id] + 0.5 * (command + 1.0) * (upper[:, joint_id] - lower[:, joint_id])
        else:
            targets = (
                self.cartesian_ik.compute(self.actions)
                if self.scripted_joint_target is None
                else self.scripted_joint_target.to(self.device)
            )
        targets = self.cfg.act_moving_average * targets + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        targets = saturate(targets, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        # MCP side-splay belongs to hand opening, not branch closure.  Keep
        # the four long digits in their neutral palm plane so the learned and
        # scripted policy cannot fold a finger behind the hand.
        for finger_id in range(2, 6):
            name = f"link_{finger_id}_mcp_spread_joint"
            if name in self.hand.joint_names:
                joint_id = self.hand.joint_names.index(name)
                targets[:, joint_id] = self.hand.data.default_joint_pos[:, joint_id]
        self.cur_targets[:] = targets
        # Keep IK targets morphology-specific. Four-finger coordination is
        # measured by the palm-coupling terms below; rewriting targets here
        # prevents selected fingers from independently reaching the branch.
        self.prev_targets[:] = self.cur_targets
        self.hand.set_joint_position_target(self.cur_targets)

    def _apply_synchronized_long_finger_rate_limit(self):
        """Advance all four long fingers by one shared angular increment.

        A digit that is close to its requested target naturally stops early;
        the remaining digits continue at the shared trajectory rate.
        """
        if not self.long_finger_joint_groups:
            return
        max_step = self.cfg.long_finger_joint_speed_rad_s * self.cfg.sim.dt * self.cfg.decimation
        # Synchronize homologous joints only. MCP, PIP and DIP need different
        # rates for Cartesian IK; flattening all of them into one group
        # destroyed the required joint ratios and made branch closure stall.
        for homologous_ids in zip(*self.long_finger_joint_groups):
            joint_ids = list(homologous_ids)
            desired = self.cur_targets[:, joint_ids]
            previous = self.prev_targets[:, joint_ids]
            remaining = desired - previous
            active = remaining.abs() > 1.0e-6
            active_count = active.sum(dim=-1, keepdim=True).clamp_min(1)
            mean_remaining = (
                torch.where(active, remaining.abs(), torch.zeros_like(remaining)).sum(
                    dim=-1, keepdim=True
                )
                / active_count
            )
            shared_step = torch.clamp(mean_remaining, 0.0, max_step)
            signed_step = torch.sign(remaining) * torch.minimum(remaining.abs(), shared_step)
            self.cur_targets[:, joint_ids] = previous + signed_step
        # Shared before individual target saturation; zero denotes perfectly matched command speed.
        self.long_finger_velocity_spread.zero_()

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        canonical_pos, canonical_vel = canonical_joint_observation(
            self.hand, self.canonical_joint_names, self.hand_dof_pos, self.hand_dof_vel,
            self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )
        obs = torch.cat(
            (
                canonical_pos,
                canonical_vel,
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.branch_pose,
                self.cartesian_ik.morphology_descriptor(),
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        fingertip_forces = torch.norm(self.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1)
        thumb_contact = fingertip_forces[:, 0] >= self.cfg.branch_contact_force_threshold
        long_finger_contacts = fingertip_forces[:, 1:] >= self.cfg.branch_contact_force_threshold
        other_finger_contact = long_finger_contacts.sum(dim=-1) >= self.cfg.min_long_finger_contacts
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
        self.extras["log"]["branch_long_finger_contact_count"] = long_finger_contacts.sum(dim=-1).float().mean()
        self.extras["log"]["branch_qualified_rate"] = qualified.float().mean()
        self.extras["log"]["branch_hold_steps"] = self.branch_success_streak.float().mean()
        self.extras["log"]["branch_finger_action_spread"] = (
            self.finger_action_scores[:, 1:].max(dim=-1).values - self.finger_action_scores[:, 1:].min(dim=-1).values
        ).mean()
        self.extras["log"]["branch_long_finger_velocity_spread"] = self.long_finger_velocity_spread.mean()
        self.extras["log"]["branch_long_finger_joint_velocity_spread"] = (
            self.long_finger_joint_velocity_spread.mean()
        )
        return self.cfg.success_reward * just_succeeded.float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.branch_success_streak >= self.cfg.branch_success_hold_steps
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)

        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (dof_pos_noise + 1.0)
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta
        dof_pos = torch.maximum(torch.minimum(dof_pos, self.hand_dof_upper_limits[env_ids]), self.hand_dof_lower_limits[env_ids])
        dof_vel = torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device)

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        # Place the branch *between* the thumb and long-finger groups.  A
        # position under the long fingers alone makes both sides approach from
        # the same direction and cannot represent an oppositional grasp.
        self.sim.forward()
        self.scene.update(dt=0.0)
        branch_state = self.branch.data.default_root_state[env_ids].clone()
        branch_state[:, 0:3], branch_state[:, 3:7] = self._find_clear_branch_pose(env_ids)
        branch_state[:, 7:] = 0.0
        self.branch.write_root_state_to_sim(branch_state, env_ids)
        self.branch_success_streak[env_ids] = 0
        self.long_finger_velocity_spread[env_ids] = 0.0
        self.cartesian_ik.reset(env_ids)

        self._compute_intermediate_values()
        self._update_long_finger_joint_velocity()
        self.long_finger_joint_velocity_scores[env_ids] = 0.0
        self.long_finger_joint_velocity_spread[env_ids] = 0.0
        relative_pos, relative_quat = self._branch_relative_pose()
        self.previous_branch_relative_pos[env_ids] = relative_pos[env_ids]
        self.previous_branch_relative_quat[env_ids] = relative_quat[env_ids]

    def _find_clear_branch_pose(self, env_ids: Sequence[int]):
        """Choose a morphology-specific branch pose without initial overlap.

        The object is searched in the local palm plane around the two-sided
        pinch point.  All surviving collision meshes are included in the
        geometry gate, so a candidate entering the palm or a proximal phalanx
        is rejected while fingertip contact remains allowed.
        """
        base = self._compute_proximal_branch_point(env_ids)
        orientation = self._branch_axis_orientation(env_ids)
        root_quat = self.hand.data.root_quat_w[env_ids]
        local_offsets = torch.tensor(
            [(y, z, 0.0) for y in (-0.016, -0.008, 0.0, 0.008, 0.016)
             for z in (-0.016, -0.008, 0.0, 0.008, 0.016)],
            dtype=torch.float32, device=self.device,
        )
        candidates = base[:, None, :] + quat_apply(
            root_quat[:, None, :].expand(-1, local_offsets.shape[0], -1),
            local_offsets[None].expand(len(env_ids), -1, -1),
        )
        body_endpoints, body_radii = self._world_collision_capsules(env_ids)
        local_axis = orientation.new_tensor((0.0, 0.0, 1.0)).unsqueeze(0).expand(len(env_ids), -1)
        branch_axis = quat_apply(orientation, local_axis)
        half_length = float(self.cfg.branch_cfg.spawn.height) * 0.5
        best_pos, best_cost = base.clone(), torch.full((len(env_ids),), float("inf"), device=self.device)
        for index in range(candidates.shape[1]):
            center = candidates[:, index]
            a, b = center - half_length * branch_axis, center + half_length * branch_axis
            clearance = self._segment_clearances(a, b, body_endpoints, body_radii, float(self.cfg.branch_cfg.spawn.radius))
            worst = clearance.min(dim=1).values
            offset_norm = torch.linalg.vector_norm(center - base, dim=-1)
            # The mesh-to-capsule envelope is conservative. Allow 1 mm of
            # envelope uncertainty, but strongly penalize larger penetration.
            # Capsule envelopes overestimate irregular convex meshes by a few
            # millimetres. Allow a bounded 6 mm approximation margin, while
            # still rejecting deeper candidate interpenetrations.
            cost = offset_norm + torch.relu(-0.006 - worst) * 100.0
            update = cost < best_cost
            best_pos = torch.where(update[:, None], center, best_pos)
            best_cost = torch.where(update, cost, best_cost)
        return best_pos, orientation

    def _load_collision_capsules(self):
        """Read actual URDF collision meshes as conservative capsules."""
        asset = Path(self.cfg.robot_cfg.spawn.asset_path)
        root = ET.parse(asset).getroot()
        runtime_bodies = set(self.hand.body_names)
        parents, transforms, joint_types = {}, {}, {}
        import numpy as np
        for joint in root.findall("joint"):
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            origin = joint.find("origin")
            xyz = np.asarray(list(map(float, origin.get("xyz", "0 0 0").split()))) if origin is not None else np.zeros(3)
            r, p, y = (list(map(float, origin.get("rpy", "0 0 0").split())) if origin is not None else (0., 0., 0.))
            rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
            ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
            rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
            transform = np.eye(4)
            transform[:3,:3], transform[:3,3] = rz @ ry @ rx, xyz
            parents[child], transforms[child], joint_types[child] = parent, transform, joint.get("type")
        result = {}
        for link in root.findall("link"):
            collision = link.find("collision")
            mesh_node = None if collision is None else collision.find("geometry/mesh")
            if mesh_node is None:
                continue
            mesh_path = Path(mesh_node.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = asset.parent / mesh_path
            raw = mesh_path.read_bytes()
            count = int.from_bytes(raw[80:84], "little")
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            vertices = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
            xyz_scale = np.asarray(list(map(float, mesh_node.get("scale", "1 1 1").split())))
            vertices = vertices * xyz_scale
            local_origin = np.zeros(3, dtype=float)
            origin = collision.find("origin")
            if origin is not None:
                local_origin = np.asarray(list(map(float, origin.get("xyz", "0 0 0").split())))
            z0, z1 = float(vertices[:, 2].min()), float(vertices[:, 2].max())
            radius = float(np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2).max())
            points = [np.r_[local_origin + (0, 0, z0), 1.0], np.r_[local_origin + (0, 0, z1), 1.0]]
            current, seen = link.get("name"), set()
            while current not in runtime_bodies:
                if current in seen or current not in parents:
                    points = []
                    break
                seen.add(current)
                if joint_types[current] != "fixed":
                    points = []
                    break
                transform = transforms[current]
                points = [transform @ point for point in points]
                current = parents[current]
            if points:
                result.setdefault(current, []).append((np.asarray(points[0][:3]), np.asarray(points[1][:3]), radius))
        return result

    def _world_collision_capsules(self, env_ids):
        endpoints, radii = [], []
        for name in self.hand.body_names:
            if name not in self._branch_collision_capsules:
                continue
            body_id = self.hand.body_names.index(name)
            if body_id in self.finger_bodies:
                continue
            pos = self.hand.data.body_pos_w[env_ids, body_id]
            quat = self.hand.data.body_quat_w[env_ids, body_id]
            for local_a, local_b, radius in self._branch_collision_capsules[name]:
                local_a_t = torch.tensor(local_a, dtype=pos.dtype, device=self.device)
                local_b_t = torch.tensor(local_b, dtype=pos.dtype, device=self.device)
                # quat_apply is batched here: expand each local endpoint to one
                # vector per environment before applying the body orientations.
                local_a_batch = local_a_t.unsqueeze(0).expand_as(pos)
                local_b_batch = local_b_t.unsqueeze(0).expand_as(pos)
                endpoints.append(torch.stack((pos + quat_apply(quat, local_a_batch),
                                               pos + quat_apply(quat, local_b_batch)), dim=1))
                radii.append(radius)
        if not endpoints:
            raise RuntimeError("No collision meshes matched the active morphology bodies")
        return torch.cat(endpoints, dim=1).reshape(len(env_ids), -1, 2, 3), torch.tensor(radii, device=self.device)

    @staticmethod
    def _segment_clearances(a, b, body_endpoints, body_radii, branch_radius):
        c, d = body_endpoints[:, :, 0], body_endpoints[:, :, 1]
        u, v, w = b[:, None] - a[:, None], d - c, a[:, None] - c
        uu = (u*u).sum(-1).clamp_min(1e-12)
        vv = (v*v).sum(-1).clamp_min(1e-12)
        uv, uw, vw = (u*v).sum(-1), (u*w).sum(-1), (v*w).sum(-1)
        det = uu*vv-uv*uv
        safe_det = det.clamp_min(1e-12)
        s = (uv*vw-vv*uw)/safe_det
        t = (uu*vw-uv*uw)/safe_det

        def squared_distance(s_param, t_param):
            delta = w+s_param[..., None]*u-t_param[..., None]*v
            return (delta*delta).sum(dim=-1)

        interior = squared_distance(s, t)
        interior = torch.where((det > 1e-12) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1),
                               interior, torch.full_like(interior, float("inf")))
        zeros = torch.zeros_like(vv)
        ones = torch.ones_like(vv)
        distance_sq = torch.minimum(
            interior,
            torch.minimum(
                torch.minimum(squared_distance(zeros, (vw/vv).clamp(0, 1)),
                              squared_distance(ones, ((vw+uv)/vv).clamp(0, 1))),
                torch.minimum(squared_distance((-uw/uu).clamp(0, 1), zeros),
                              squared_distance(((uv-uw)/uu).clamp(0, 1), ones)),
            ),
        )
        distance = distance_sq.clamp_min(0).sqrt()
        return distance - branch_radius - body_radii[None]

    def _compute_intermediate_values(self):
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.unsqueeze(1)
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        self.branch_pos = self.branch.data.root_pos_w - self.scene.env_origins
        self.branch_rot = self.branch.data.root_quat_w
        self.branch_pose = torch.cat((self.branch_pos, self.branch_rot), dim=-1)

    def _compute_proximal_branch_point(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Place the branch in the actual two-sided fingertip workspace.

        The proximal-only center could be several centimeters away from the
        thumb/long-finger pinch plane for this morphology.  Use the midpoint
        of the thumb tip and the long-finger tip sheet, then move a few
        millimeters toward the thumb. Full-shape clearance needs validation.
        """
        tip_positions = self.hand.data.body_pos_w[env_ids][:, self.finger_bodies]
        thumb = tip_positions[:, :1]
        long_center = self.hand.data.body_pos_w[env_ids][:, self.proximal_branch_body_ids].mean(dim=1)
        pinch_center = 0.5 * (thumb[:, 0] + long_center)
        opposition = long_center - thumb[:, 0]
        opposition = opposition / torch.linalg.vector_norm(opposition, dim=-1, keepdim=True).clamp_min(1.0e-6)
        pinch_center = pinch_center + 0.004 * opposition
        # A small thumbward bias is not a collision-clearance guarantee;
        # validate the generated shape's full collision geometry separately.
        opposition = thumb[:, 0] - long_center
        opposition = opposition / torch.linalg.vector_norm(opposition, dim=-1, keepdim=True).clamp_min(1.0e-6)
        center = pinch_center + 0.001 * opposition
        # Calibration is expressed in the hand frame so the native training
        # reset and scripted demonstration use exactly the same placement.
        default_offset = "-0.037,-0.046,0.001" if os.environ.get("EVOLUTION_BRANCH_BC_MODE") == "1" else "0,0,0"
        offset = tuple(float(value) for value in os.environ.get(
            "EVOLUTION_BRANCH_RESET_OFFSET_LOCAL", default_offset
        ).split(","))
        if len(offset) != 3:
            raise ValueError("EVOLUTION_BRANCH_RESET_OFFSET_LOCAL needs three values")
        local = torch.tensor(offset, dtype=center.dtype, device=self.device).expand(len(env_ids), -1)
        return center + quat_apply(self.hand.data.root_quat_w[env_ids], local)

    def _branch_axis_orientation(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Align the cylinder along the long-finger row, not along finger reach."""
        proximal_bodies = self.hand.data.body_pos_w[env_ids][:, self.proximal_branch_body_ids]
        axis = proximal_bodies[:, -1] - proximal_bodies[:, 0]
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-6)
        # Quaternion rotating the cylinder's local +Z axis onto the finger row.
        z_axis = torch.zeros_like(axis)
        z_axis[:, 2] = 1.0
        cross = torch.cross(z_axis, axis, dim=-1)
        dot = (z_axis * axis).sum(dim=-1, keepdim=True)
        quat = torch.cat((1.0 + dot, cross), dim=-1)
        opposite = quat[:, 0].abs() < 1.0e-5
        if opposite.any():
            quat[opposite] = torch.tensor((0.0, 1.0, 0.0, 0.0), device=self.device)
        return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def _update_long_finger_joint_velocity(self):
        """Measure achieved, rather than commanded, four-finger closing speed."""
        normalized_joint_pos = unscale(
            self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )
        scores = torch.stack(
            [
                normalized_joint_pos[:, joint_ids].mean(dim=-1)
                for joint_ids in self.long_finger_joint_groups
            ],
            dim=-1,
        )
        self.long_finger_joint_scores = scores
        self.long_finger_joint_velocity_scores = scores - self.previous_long_finger_joint_scores
        self.long_finger_joint_velocity_spread = (
            self.long_finger_joint_velocity_scores.max(dim=-1).values
            - self.long_finger_joint_velocity_scores.min(dim=-1).values
        )
        self.previous_long_finger_joint_scores = scores

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
