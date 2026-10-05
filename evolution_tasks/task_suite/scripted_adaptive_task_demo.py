"""Morphology-adaptive scripted demonstrations for BranchGrasp, Forage and Strike."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def main(task: str) -> None:
    parser = argparse.ArgumentParser(description=f"Record a morphology-adaptive {task} demonstration.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--lineage_json", help="Optional evolution result JSON containing lineage data.")
    parser.add_argument("--individual_key", help="Individual key in lineage_json, for example 15_0.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_video_steps", type=int, default=90, help="Minimum rendered frames; 90 at 30 fps is 3 seconds.")
    parser.add_argument("--preflight", action="store_true", help="Run physical success checks without rendering video.")
    parser.add_argument("--training_scene", action="store_true", help="Use native task reset without scripted object or hand state writes.")
    parser.add_argument("--branch_policy_demo", action="store_true", help="Use the Branch policy's 20 action channels without joint overrides.")
    parser.add_argument("--replay_actions_json", help="Forage diagnostic: replay recorded policy actions in a fresh task reset.")
    parser.add_argument("--direct_envelope_closure", action=argparse.BooleanOptionalAction, default=True, help="Retain the joint target used to generate the grasp envelope.")
    parser.add_argument("--max_physical_candidates", type=int, default=36)
    parser.add_argument("--strike_thumb_bias", type=float, default=0.0, help="Diagnostic tool-position bias toward the closed thumb, in metres.")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    if bool(args.lineage_json) != bool(args.individual_key):
        parser.error("--lineage_json and --individual_key must be supplied together")
    if args.branch_policy_demo and (task != "branch" or not args.training_scene or os.environ.get("EVOLUTION_BRANCH_BC_MODE") != "1"):
        parser.error("--branch_policy_demo requires Branch, --training_scene and EVOLUTION_BRANCH_BC_MODE=1")
    args.enable_cameras = not args.preflight
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    def _bc_array(value):
        if isinstance(value, dict):
            return np.concatenate([_bc_array(value[key]).reshape(-1) for key in sorted(value)])
        if isinstance(value, (tuple, list)):
            return np.concatenate([_bc_array(item).reshape(-1) for item in value])
        return np.asarray(value, dtype=np.float32).reshape(-1)
    from isaaclab.utils.math import quat_apply, quat_conjugate

    import isaaclab_tasks  # noqa: F401

    def prepare_hand(output_root: Path):
        code_root = Path(os.environ.get("EVOLUTION_CODE_ROOT", "/home/zjh/Evolution_PC"))
        sys.path.insert(0, str(code_root / "Isaaclab_other"))
        from code_to_urdf import generate_urdf_from_dict
        from human_hand_agent import initial_agent_hand
        from isaaclab_tool import parse_urdf_and_generate_articulation_cfg

        if args.lineage_json:
            payload = json.loads(Path(args.lineage_json).read_text(encoding="utf-8"))
            hand = payload["lineage"][args.individual_key]["urdf_info"]
            hand = dict(hand)
            hand["agent_code"] = f"adaptive_{task}_{args.individual_key}"
        else:
            hand = dict(initial_agent_hand)
            hand["agent_code"] = f"adaptive_{task}_human"
        urdf = output_root / "morphology" / "urdf" / "current_agent.urdf"
        urdf.parent.mkdir(parents=True, exist_ok=True)
        generate_urdf_from_dict(hand, str(output_root / "morphology" / "meshes"), str(urdf))
        cfg_path = output_root / "morphology" / "current_hand_cfg.py"
        parse_urdf_and_generate_articulation_cfg(str(urdf), str(urdf), str(cfg_path))
        spec = importlib.util.spec_from_file_location(f"adaptive_{task}_hand_cfg", cfg_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load generated hand config: {cfg_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        available_joints = set(module.MORPHOLOGY_CONTRACT["all_actuated_joints"])
        fingertip_names = []
        for finger_id in range(1, 6):
            body_index = None
            for index, joint_name in enumerate((
                f"link_0_0_to_link_{finger_id}_0",
                f"link_{finger_id}_0_to_link_{finger_id}_1",
                f"link_{finger_id}_1_to_link_{finger_id}_2",
                f"link_{finger_id}_2_to_link_{finger_id}_3",
            )):
                if joint_name in available_joints:
                    body_index = index
            if body_index is None:
                raise RuntimeError(f"Finger {finger_id} has no remaining actuated body")
            fingertip_names.append(f"link_{finger_id}_{body_index}")
        return module.CURRENT_HAND_CFG, fingertip_names

    def u8(frame):
        frame = frame[0] if isinstance(frame, (tuple, list)) else frame
        return np.asarray(frame).astype(np.uint8)

    def cartesian_action(raw, world_targets: torch.Tensor, action_dim: int) -> torch.Tensor:
        tip_pos = raw.cartesian_ik.fingertip_positions_world()
        delta_world = world_targets - tip_pos
        local_quat = quat_conjugate(raw.hand.data.root_quat_w).unsqueeze(1).expand(-1, 5, -1)
        delta_local = quat_apply(local_quat, delta_world)
        action = torch.zeros((1, action_dim), device=raw.device)
        action[:, :15] = torch.clamp(delta_local / 0.005, -1.0, 1.0).reshape(1, 15)
        return action

    def axis_to_quaternion(axis: torch.Tensor) -> torch.Tensor:
        """Quaternion rotating the cylinder's local +Z axis onto ``axis``."""
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-6)
        z_axis = torch.zeros_like(axis)
        z_axis[:, 2] = 1.0
        cross = torch.cross(z_axis, axis, dim=-1)
        dot = (z_axis * axis).sum(dim=-1, keepdim=True)
        quat = torch.cat((1.0 + dot, cross), dim=-1)
        return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def morphology_closure_target(raw, fraction: float = 0.82, thumb_angle: float = -1.15) -> torch.Tensor:
        """Create a limit-safe closure target for the loaded morphology."""
        limits = raw.hand.root_physx_view.get_dof_limits().to(raw.device)
        lower, upper = limits[..., 0], limits[..., 1]
        target = raw.hand.data.joint_pos.clone()
        for joint_id, name in enumerate(raw.hand.joint_names):
            if name == "link_1_thumb_spread_joint":
                target[:, joint_id] = torch.clamp(
                    torch.full_like(target[:, joint_id], thumb_angle), lower[:, joint_id], upper[:, joint_id]
                )
            elif "mcp_spread_joint" in name:
                target[:, joint_id] = 0.0
            elif (
                name.startswith("link_0_0_to_link_")
                or "_0_to_link_" in name
                or "_1_to_link_" in name
            ):
                target[:, joint_id] = lower[:, joint_id] + fraction * (upper[:, joint_id] - lower[:, joint_id])
        return torch.clamp(target, lower, upper)

    def _set_joint_state(raw, joint_state: torch.Tensor) -> None:
        """Evaluate a candidate pose without advancing the task episode."""
        zero_velocity = torch.zeros_like(joint_state)
        raw.hand.write_joint_state_to_sim(joint_state, zero_velocity)
        raw.hand.set_joint_position_target(joint_state)
        raw.sim.forward()
        raw.scene.update(dt=0.0)
        raw._compute_intermediate_values()

    def _circumcenter(points: np.ndarray):
        """Return the 2-D circumcenter and radius, or None for collinear points."""
        a, b, c = points
        matrix = 2.0 * np.array((b - a, c - a), dtype=np.float64)
        rhs = np.array((b @ b - a @ a, c @ c - a @ a), dtype=np.float64)
        if abs(np.linalg.det(matrix)) < 1.0e-7:
            return None
        center = np.linalg.solve(matrix, rhs)
        return center, float(np.linalg.norm(center - a))

    def choose_adaptive_pregasp(raw, task: str, initial_joint_pos: torch.Tensor, cfg):
        """Choose a legal closure and object pose from the loaded morphology.

        The object is placed against the measured closed fingertip envelope,
        rather than against a nominal hand coordinate that changes with
        morphology evolution.
        """
        far_state = None
        if task == "branch":
            far_state = raw.branch.data.default_root_state[:1].clone()
            far_state[:, 2] += 1.0
            raw.branch.write_root_state_to_sim(far_state)
            # Search for an actually compressive fingertip envelope.  The
            # previous radius-plus-pad target was only tangent and generated
            # about 0.03 N, far below the 1 N task threshold.
            target_radius = float(cfg.branch_cfg.spawn.radius) - 0.005
            plane = "yz"
        else:
            far_state = raw.cone.data.default_root_state[:1].clone()
            far_state[:, 2] += 1.0
            raw.cone.write_root_state_to_sim(far_state)
            target_radius = float(cfg.Cone_cfg.spawn.size[0]) * 0.5 + 0.006
            plane = "xy"

        best = None
        fractions = (0.35, 0.50, 0.65, 0.80, 0.92)
        thumb_angles = (-0.80, -1.15, -1.50, -1.80)
        for fraction in fractions:
            for thumb_angle in thumb_angles:
                candidate = morphology_closure_target(raw, fraction, thumb_angle)
                # The object must occupy the envelope reached after closure.
                # Using the open candidate pose placed it several centimetres
                # away from the final fingertips on evolved morphologies.
                _set_joint_state(raw, candidate)
                tips = raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().numpy()
                for pair in itertools.combinations(range(1, 5), 2):
                    ids = (0, pair[0], pair[1])
                    points = tips[list(ids)]
                    points_2d = points[:, (1, 2)] if plane == "yz" else points[:, (0, 1)]
                    circle = _circumcenter(points_2d)
                    if circle is None:
                        continue
                    center_2d, radius = circle
                    # Prefer a three-point envelope close to the existing
                    # object radius, without rewarding deep penetration.
                    score = abs(radius - target_radius) + max(0.0, target_radius - radius) * 0.35
                    if best is not None and score >= best[0]:
                        continue
                    if plane == "yz":
                        position = np.array((points[:, 0].mean(), center_2d[0], center_2d[1]))
                    else:
                        position = np.array((center_2d[0], center_2d[1], np.median(points[:, 2])))
                    best = (score, candidate.clone(), position, fraction, thumb_angle, ids, radius)

        if best is None:
            raise RuntimeError(f"Could not find a non-collinear adaptive {task} pre-grasp envelope")
        _set_joint_state(raw, initial_joint_pos)
        # Place the object at the tangent envelope, then command a slightly
        # deeper legal closure.  Using one pose for both operations only
        # produced grazing contacts (about 0.03 N) and could never satisfy the
        # sustained-force gates.
        closure_fraction = min(0.98, best[3] + (0.18 if task == "branch" else 0.08))
        closure_thumb_angle = best[4] - (0.20 if task == "branch" else 0.08)
        closure_target = morphology_closure_target(raw, closure_fraction, closure_thumb_angle)
        return closure_target, torch.tensor(best[2], dtype=torch.float32, device=raw.device).view(1, 3), {
            "closure_fraction": best[3],
            "thumb_angle_rad": best[4],
            "compression_fraction": closure_fraction,
            "compression_thumb_angle_rad": closure_thumb_angle,
            "contact_finger_ids": list(best[5]),
            "envelope_radius_m": best[6],
            "target_radius_m": target_radius,
        }

    def calibrate_physical_pregasp(env, raw, task: str, cfg, seed_position: torch.Tensor):
        """Select a morphology-specific pose using measured Isaac contacts."""
        action_dim = 20 if task == "branch" else 23
        action = torch.zeros((1, action_dim), device=raw.device)
        fractions = (0.20, 0.35, 0.50, 0.65, 0.80)
        thumb_angles = (-1.00, -1.40, -1.80, -2.08)
        offsets = (-0.010, 0.0, 0.010) if task == "branch" else (-0.015, 0.0, 0.015)
        best = None
        old_branch_hold = getattr(cfg, "branch_success_hold_steps", None)
        old_tool_hold = getattr(cfg, "tool_grasp_hold_steps", None)
        if old_branch_hold is not None:
            cfg.branch_success_hold_steps = 10000
        if old_tool_hold is not None:
            cfg.tool_grasp_hold_steps = 10000
        required_force = (
            cfg.branch_contact_force_threshold
            if task == "branch"
            else cfg.tool_finger_contact_force_threshold
        )
        threshold_reached = False
        physical_candidates = []
        try:
            for fraction, thumb_angle in itertools.product(fractions, thumb_angles):
                candidate = morphology_closure_target(raw, fraction, thumb_angle)
                compression = morphology_closure_target(
                    raw, min(0.98, fraction + 0.55), thumb_angle - 0.30
                )
                # Place the object from the final reachable envelope.  The
                # pre-grasp pose is used only as the start of the trajectory;
                # computing the center from that open pose made the selected
                # long fingers converge on different cylinder cross-sections.
                geometry_pose = compression
                _set_joint_state(raw, geometry_pose)
                tips = raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().numpy()
                if task == "branch":
                    # The branch radius is 22 mm and the fingertip collision
                    # capsule is about 8 mm. Candidate geometry therefore
                    # uses center-to-center tangency near 30 mm; asking for a
                    # 10 mm radius deeply embeds the finger origins and PhysX
                    # resolves it by ejecting the digits sideways.
                    target_radius = float(cfg.branch_cfg.spawn.radius) + 0.008
                    proximal = raw.hand.data.body_pos_w[
                        0, raw.proximal_branch_body_ids
                    ].detach().cpu().numpy()
                    axis = proximal[-1] - proximal[0]
                    axis = axis / max(np.linalg.norm(axis), 1.0e-8)
                    reference = np.array((0.0, 0.0, 1.0))
                    if abs(float(axis @ reference)) > 0.9:
                        reference = np.array((0.0, 1.0, 0.0))
                    basis_u = np.cross(axis, reference)
                    basis_u = basis_u / max(np.linalg.norm(basis_u), 1.0e-8)
                    basis_v = np.cross(axis, basis_u)
                else:
                    target_radius = float(cfg.Cone_cfg.spawn.size[0]) * 0.5 + 0.006
                    basis_u = np.array((1.0, 0.0, 0.0))
                    basis_v = np.array((0.0, 1.0, 0.0))
                envelope = []
                for pair in itertools.combinations(range(1, 5), 2):
                    ids = (0, pair[0], pair[1])
                    points = tips[list(ids)]
                    points_2d = np.stack((points @ basis_u, points @ basis_v), axis=-1)
                    circle = _circumcenter(points_2d)
                    if circle is not None:
                        center_2d, radius = circle
                        axial_coordinate = float(seed_position[0].detach().cpu().numpy() @ axis) \
                            if task == "branch" else float(points[:, 2].mean())
                        center = center_2d[0] * basis_u + center_2d[1] * basis_v
                        if task == "branch":
                            center = center + axial_coordinate * axis
                        else:
                            center[2] = axial_coordinate
                        envelope.append((abs(radius - target_radius), center, ids, radius))
                if not envelope:
                    continue
                envelope.sort(key=lambda item: item[0])
                geometry_error, base_position, ids, radius = envelope[0]
                for offset_a, offset_b in itertools.product(offsets, offsets):
                    position = torch.tensor(base_position, dtype=torch.float32, device=raw.device).view(1, 3)
                    position = position.clone()
                    position += torch.tensor(
                        offset_a * basis_u + offset_b * basis_v,
                        dtype=torch.float32,
                        device=raw.device,
                    ).view(1, 3)
                    if task == "strike" and args.strike_thumb_bias:
                        toward_thumb = torch.tensor(tips[0, :2], device=raw.device) - position[0, :2]
                        toward_thumb /= torch.linalg.vector_norm(toward_thumb).clamp_min(1e-6)
                        position[:, :2] += args.strike_thumb_bias * toward_thumb
                    selected = tips[list(ids)]
                    thumb_vector = selected[0] - position[0].detach().cpu().numpy()
                    long_vectors = selected[1:] - position[0].detach().cpu().numpy()
                    pair_oppositions = [
                        -float(
                            np.dot(thumb_vector, vector)
                            / max(np.linalg.norm(thumb_vector) * np.linalg.norm(vector), 1.0e-8)
                        )
                        for vector in long_vectors
                    ]
                    opposition = min(pair_oppositions)
                    if opposition < 0.20:
                        continue
                    physical_candidates.append(
                        (
                            geometry_error + 0.01 * (1.0 - opposition)
                            + 0.05 * (offset_a * offset_a + offset_b * offset_b),
                            candidate.clone(),
                            compression.clone(),
                            position.clone(),
                            fraction,
                            thumb_angle,
                            axis.copy() if task == "branch" else None,
                            ids,
                            radius,
                            opposition,
                        )
                    )

            # Geometry is cheap; PhysX contact probes are not. Validate only
            # the best diverse envelopes instead of simulating every grid point.
            physical_candidates.sort(key=lambda item: item[0])
            for candidate_index, candidate_data in enumerate(physical_candidates[:args.max_physical_candidates]):
                candidate_started = time.monotonic()
                print(f"[PREFLIGHT] Candidate {candidate_index + 1}/{min(args.max_physical_candidates, len(physical_candidates))}", flush=True)
                (
                    _, candidate, compression, position, fraction, thumb_angle,
                    axis, ids, radius, opposition,
                ) = candidate_data
                env.reset(seed=args.seed)
                raw.scripted_joint_target = None
                fixed_target = compression.clone()
                selected_fingers = (1, ids[1] + 1, ids[2] + 1)
                hard_limits = raw.hand.root_physx_view.get_dof_limits().to(raw.device)
                for joint_index, joint_name in enumerate(raw.hand.joint_names):
                    if "spread_joint" in joint_name:
                        continue
                    if any(
                        joint_name.startswith(f"link_{finger_id}_")
                        or f"to_link_{finger_id}_" in joint_name
                        for finger_id in selected_fingers
                    ):
                        fixed_target[:, joint_index] += 0.12 * (
                            hard_limits[:, joint_index, 1] - fixed_target[:, joint_index]
                        )
                fixed_target = torch.clamp(
                    fixed_target, hard_limits[..., 0], hard_limits[..., 1]
                )
                envelope_joint_target = fixed_target.clone()
                if task == "branch":
                    state = raw.branch.data.default_root_state[:1].clone()
                    state[:, :3] = position
                    state[:, 3:7] = axis_to_quaternion(
                        torch.tensor(axis, dtype=torch.float32, device=raw.device).view(1, 3)
                    )
                    state[:, 7:] = 0.0
                    # Solve the morphology-specific contact pose in free
                    # space first. Contact impulses otherwise deflect the
                    # fingers while IK is still trying to discover the pose.
                    far_state = state.clone()
                    far_state[:, 2] += 1.0
                    raw.branch.write_root_state_to_sim(far_state)
                    _set_joint_state(raw, candidate)
                    raw.prev_targets[:] = candidate
                    raw.cur_targets[:] = candidate
                    axis_tensor = torch.tensor(axis, dtype=torch.float32, device=raw.device)
                    initial_tips = raw.cartesian_ik.fingertip_positions_world()
                    relative = initial_tips - position.unsqueeze(1)
                    axial = (relative * axis_tensor).sum(dim=-1, keepdim=True) * axis_tensor
                    radial = relative - axial
                    radial_direction = radial / torch.linalg.vector_norm(
                        radial, dim=-1, keepdim=True
                    ).clamp_min(1.0e-5)
                    surface_radius = float(cfg.branch_cfg.spawn.radius) + 0.006
                    free_targets = initial_tips.clone()
                    contact_targets = (
                        position.unsqueeze(1) + axial + radial_direction * surface_radius
                    )
                    selected_ids = (0, ids[1], ids[2])
                    free_targets[:, selected_ids] = contact_targets[:, selected_ids]
                    raw.scripted_joint_target = None
                    for _ in range(0 if args.direct_envelope_closure else 120):
                        env.step(cartesian_action(raw, free_targets, action_dim))
                    fixed_target = raw.prev_targets.clone()
                    free_target = fixed_target.clone()
                    for joint_index, joint_name in enumerate(raw.hand.joint_names):
                        if "spread_joint" in joint_name:
                            continue
                        if any(
                            joint_name.startswith(f"link_{finger_id}_")
                            or f"to_link_{finger_id}_" in joint_name
                            for finger_id in selected_fingers
                        ):
                            fixed_target[:, joint_index] += 0.50 * (
                                free_target[:, joint_index] - candidate[:, joint_index]
                            )
                    fixed_target = torch.clamp(
                        fixed_target, hard_limits[..., 0], hard_limits[..., 1]
                    )

                    if args.direct_envelope_closure:
                        fixed_target = envelope_joint_target
                    env.reset(seed=args.seed)
                    raw.branch.write_root_state_to_sim(state)
                    _set_joint_state(raw, candidate)
                    raw.prev_targets[:] = candidate
                    raw.cur_targets[:] = candidate
                else:
                    state = raw.cone.data.default_root_state[:1].clone()
                    state[:, :3] = position
                    state[:, 7:] = 0.0
                    raw.cone.write_root_state_to_sim(state)
                _set_joint_state(raw, candidate)
                raw.prev_targets[:] = candidate
                raw.cur_targets[:] = candidate
                raw.scripted_joint_target = fixed_target
                measured = []
                final_forces = None
                fixed_forces = None
                fixed_joint_pos = None
                for probe_step in range(160):
                    if task == "strike":
                        raw.cone.write_root_state_to_sim(state)
                    if task != "branch" and probe_step == 64:
                        raw.scripted_joint_target = None
                    if task != "branch" and probe_step >= 64:
                        current_tips = raw.cartesian_ik.fingertip_positions_world()
                        targets = current_tips.clone()
                        if task == "branch":
                            axis_tensor = torch.tensor(axis, dtype=torch.float32, device=raw.device)
                            relative = current_tips - position.unsqueeze(1)
                            axial = (relative * axis_tensor).sum(dim=-1, keepdim=True) * axis_tensor
                            radial = relative - axial
                            radial_direction = radial / torch.linalg.vector_norm(
                                radial, dim=-1, keepdim=True
                            ).clamp_min(1.0e-5)
                            contact_radius = max(0.004, float(cfg.branch_cfg.spawn.radius) - 0.002)
                            contact_targets = position.unsqueeze(1) + axial + radial_direction * contact_radius
                        else:
                            relative = current_tips - position.unsqueeze(1)
                            radial = relative.clone()
                            radial[..., 2] = 0.0
                            radial_direction = radial / torch.linalg.vector_norm(
                                radial, dim=-1, keepdim=True
                            ).clamp_min(1.0e-5)
                            contact_radius = float(cfg.Cone_cfg.spawn.size[0]) * 0.5 - 0.002
                            contact_targets = current_tips.clone()
                            contact_targets[..., :2] = (
                                position[:, None, :2]
                                + radial_direction[..., :2] * contact_radius
                            )
                        selected_ids = (0, ids[1], ids[2])
                        targets[:, selected_ids] = contact_targets[:, selected_ids]
                        action = cartesian_action(raw, targets, action_dim)
                    else:
                        action.zero_()
                    env.step(action)
                    raw._compute_intermediate_values()
                    if task == "branch":
                        forces = torch.norm(
                            raw.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1
                        )[0]
                    else:
                        forces = raw.tool_fingertip_forces[0]
                    final_forces = forces.detach().cpu().tolist()
                    long_second = torch.topk(forces[1:], k=2).values[-1]
                    measured.append(float(torch.minimum(forces[0], long_second)))
                    if probe_step == 63:
                        fixed_forces = final_forces
                        fixed_joint_pos = raw.hand.data.joint_pos.clone()

                fixed_score = min(measured[54:64])
                ik_score = min(measured[-10:])
                use_fixed_target = fixed_score >= ik_score
                stable_score = max(fixed_score, ik_score)
                peak = max(measured)
                score = stable_score - max(0.0, peak - 8.0) * 0.1
                joint_pos = fixed_joint_pos if use_fixed_target else raw.hand.data.joint_pos
                hard_limits = raw.hand.root_physx_view.get_dof_limits().to(raw.device)
                limit_violation = bool(
                    (~torch.isfinite(joint_pos)).any()
                    or (joint_pos < hard_limits[..., 0] - 0.020).any()
                    or (joint_pos > hard_limits[..., 1] + 0.020).any()
                )
                if limit_violation:
                    score = -1.0
                released_hold_steps = None
                if task == "strike" and score >= required_force:
                    # A fixed tool can manufacture contact. The real preflight
                    # passes only if the fully dynamic tool remains grasped.
                    raw.scripted_joint_target = (fixed_target if use_fixed_target else raw.prev_targets).clone()
                    raw.tool_grasp_streak.zero_()
                    required_hold_steps = int(old_tool_hold)
                    for _ in range(max(30, required_hold_steps + 2)):
                        env.step(torch.zeros((1, action_dim), device=raw.device))
                        raw._compute_intermediate_values()
                    released_hold_steps = int(raw.tool_grasp_streak[0])
                    if released_hold_steps < required_hold_steps:
                        score = -1.0
                debug = {
                    "final_fingertip_forces_n": fixed_forces if use_fixed_target else final_forces,
                    "fixed_stable_score_n": fixed_score,
                    "ik_stable_score_n": ik_score,
                    "selected_controller": "fixed_joint_target" if task == "branch" or use_fixed_target else "cartesian_ik",
                    "contact_finger_ids": list(ids),
                    "opposition": opposition,
                    "released_hold_steps": released_hold_steps,
                    "joint_limit_violation": limit_violation,
                    "ik_error_m": raw.cartesian_ik.last_ik_error[0].detach().cpu().tolist(),
                    "ik_reachable": raw.cartesian_ik.last_ik_reachable[0].detach().cpu().tolist(),
                    "joint_target_error_rad": (
                        raw.prev_targets[0] - raw.hand.data.joint_pos[0]
                    ).detach().cpu().tolist(),
                    "joint_positions_rad": raw.hand.data.joint_pos[0].detach().cpu().tolist(),
                    "joint_limits_rad": raw.hand.root_physx_view.get_dof_limits()[0].detach().cpu().tolist(),
                }
                if task == "branch":
                    current_tips = raw.cartesian_ik.fingertip_positions_world()[0]
                    branch_center = raw.branch.data.root_pos_w[0]
                    axis_tensor = torch.tensor(axis, dtype=torch.float32, device=raw.device)
                    relative = current_tips - branch_center
                    radial = relative - (relative * axis_tensor).sum(dim=-1, keepdim=True) * axis_tensor
                    debug.update(
                        radial_distances_m=torch.linalg.vector_norm(radial, dim=-1).detach().cpu().tolist(),
                        branch_center_m=branch_center.detach().cpu().tolist(),
                        fingertip_positions_m=current_tips.detach().cpu().tolist(),
                        sensor_shape=list(raw.branch_contact_sensor.data.force_matrix_w.shape),
                        axis_world=axis.tolist(),
                    )
                else:
                    debug.update(
                        fingertip_positions_m=raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
                        tool_position_m=raw.cone.data.root_pos_w[0].detach().cpu().tolist(),
                        tool_quaternion_wxyz=raw.cone.data.root_quat_w[0].detach().cpu().tolist(),
                    )
                progress = {
                    "candidate": candidate_index + 1,
                    "elapsed_seconds": time.monotonic() - candidate_started,
                    "score_n": score, "required_force_n": float(required_force),
                    "direct_envelope_closure": args.direct_envelope_closure,
                    "fraction": fraction, "thumb_angle_rad": thumb_angle,
                    "position_m": position[0].detach().cpu().tolist(),
                    "debug": debug,
                }
                progress_path = Path(args.metrics).with_suffix(".candidates.jsonl")
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as progress_file:
                    progress_file.write(json.dumps(progress) + "\n")
                print("[PREFLIGHT] Candidate result", {key: value for key, value in progress.items() if key != "debug"}, flush=True)
                if best is None or score > best[0]:
                    selected_target = fixed_target if use_fixed_target else raw.prev_targets
                    best = (
                        score,
                        candidate.clone(),
                        selected_target.clone(),
                        position.clone(),
                        fraction,
                        thumb_angle,
                        measured,
                        debug,
                    )
                if score >= required_force:
                    threshold_reached = True
                    break
        finally:
            if old_branch_hold is not None:
                cfg.branch_success_hold_steps = old_branch_hold
            if old_tool_hold is not None:
                cfg.tool_grasp_hold_steps = old_tool_hold
        if best is None:
            raise RuntimeError(f"Physical pre-grasp calibration failed for {task}")
        if best[0] < required_force:
            raise RuntimeError(
                f"No {task} pre-grasp reached the physical contact threshold: "
                f"best={best[0]:.4f} N required={required_force:.4f} N "
                f"debug={json.dumps(best[7])}"
            )
        env.reset(seed=args.seed)
        raw.branch_success_streak.zero_() if task == "branch" else raw.tool_grasp_streak.zero_()
        return best[1], best[2], best[3], {
            "physical_score_n": best[0],
            "physical_fraction": best[4],
            "physical_thumb_angle_rad": best[5],
            "physical_probe_forces_n": best[6],
            "physical_debug": best[7],
        }

    generated_hand_cfg, fingertip_names = prepare_hand(Path(args.output).parent)
    task_import = {
        "branch": "isaaclab_tasks.evolution_tasks.task_branch_grasp",
        "forage": "isaaclab_tasks.evolution_tasks.task_forage",
        "strike": "isaaclab_tasks.evolution_tasks.task_strike",
    }[task]
    importlib.import_module(task_import)
    cfg_module = importlib.import_module({
        "branch": "isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg",
        "forage": "isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg",
        "strike": "isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg",
    }[task])
    cfg_class = {"branch": "BranchGraspEnvCfg", "forage": "ForageEnvCfg", "strike": "EvolutionStrikeEnvCfg"}[task]
    cfg = getattr(cfg_module, cfg_class)()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.seed = args.seed
    # Isaac merges each fixed link_*_3 fingertip pad into link_*_2 on import.
    # Sensors and Cartesian targets must therefore use the surviving body names.
    cfg.fingertip_body_names = fingertip_names
    cfg.robot_cfg = generated_hand_cfg.replace(prim_path=cfg.robot_cfg.prim_path).replace(init_state=cfg.robot_cfg.init_state)
    if task == "branch":
        # Keep task geometry authoritative; only pose is morphology-adaptive.
        cfg.viewer.eye, cfg.viewer.lookat = (0.24, -0.20, 0.38), (0.0, 0.0, 0.30)
    elif task == "forage":
        cfg.viewer.eye, cfg.viewer.lookat = (0.28, -0.26, 0.42), (0.0, 0.0, 0.11)
    else:
        cfg.viewer.eye, cfg.viewer.lookat = (-0.46, -0.40, 0.46), (-0.05, 0.01, 0.19)
    # Preflight must use the training morphology's drives and task physics.
    effective_drives = {
        name: {key: getattr(actuator, key) for key in (
            "stiffness", "damping", "velocity_limit_sim", "effort_limit_sim"
        )}
        for name, actuator in cfg.robot_cfg.actuators.items()
    }
    print("[PREFLIGHT] Effective drives", effective_drives, flush=True)

    env_id = {"branch": "Isaac-EvolutionHand-BranchGrasp-v0", "forage": "Isaac-EvolutionHand-Forage-v0", "strike": "Isaac-EvolutionHand-Strike-v0"}[task]
    env = gym.make(env_id, cfg=cfg, render_mode=None if args.preflight else "rgb_array")
    raw = env.unwrapped
    env.reset(seed=args.seed)
    raw._compute_intermediate_values()
    initial_joint_pos = raw.hand.data.joint_pos.clone()
    adaptive_geometry = None
    closure_target = morphology_closure_target(raw)
    physical_pregrasp_target = initial_joint_pos.clone()
    if task in ("branch", "strike") and not args.training_scene:
        closure_target, adaptive_object_position, adaptive_geometry = choose_adaptive_pregasp(
            raw, task, initial_joint_pos, cfg
        )
        physical_pregrasp_target, closure_target, adaptive_object_position, physical_geometry = calibrate_physical_pregasp(
            env, raw, task, cfg, adaptive_object_position
        )
        adaptive_geometry.update(physical_geometry)
    if task == "branch" and not args.training_scene:
        branch_state = raw.branch.data.default_root_state[:1].clone()
        branch_state[:, :3] = adaptive_object_position
        selected_axis = physical_geometry.get("physical_debug", {}).get("axis_world") if physical_geometry else None
        if selected_axis is None:
            branch_state[:, 3:7] = raw._branch_axis_orientation([0])
        else:
            branch_state[:, 3:7] = axis_to_quaternion(
                torch.tensor(selected_axis, dtype=torch.float32, device=raw.device).view(1, 3)
            )
        branch_state[:, 7:] = 0.0
        raw.branch.write_root_state_to_sim(branch_state)
    elif task == "strike" and not args.training_scene:
        # Replay the measured free-tool equilibrium, not the actuator target.
        grasp_debug = physical_geometry["physical_debug"]
        tool_state = raw.cone.data.default_root_state[:1].clone()
        tool_state[:, :3] = torch.tensor(grasp_debug["tool_position_m"], device=raw.device)
        tool_state[:, 3:7] = torch.tensor(grasp_debug["tool_quaternion_wxyz"], device=raw.device)
        tool_state[:, 7:] = 0.0
        raw.cone.write_root_state_to_sim(tool_state)
        # Strike starts from the physically calibrated closed grasp. The tool
        # remains fully dynamic; no root-state write is used to hold it.
        _set_joint_state(raw, torch.tensor(grasp_debug["joint_positions_rad"], device=raw.device).unsqueeze(0))
        initial_joint_pos = closure_target.clone()
        raw.prev_targets[:] = initial_joint_pos
        raw.cur_targets[:] = initial_joint_pos
    raw.sim.forward()
    raw.scene.update(dt=0.0)
    if not args.preflight:
        from isaacsim.core.utils.viewports import set_camera_view
        if task == "branch":
            raw._compute_intermediate_values()
            branch_center = raw.branch.data.root_pos_w[0]
            eye = branch_center + torch.tensor((0.28, -0.34, 0.18), device=raw.device)
            set_camera_view(eye=tuple(eye.detach().cpu().tolist()), target=tuple(branch_center.detach().cpu().tolist()), camera_prim_path="/OmniverseKit_Persp")
        else:
            set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat, camera_prim_path="/OmniverseKit_Persp")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = None if args.preflight else imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
    history = []
    bc_observations = []
    bc_actions = []
    replay_actions = None
    if args.replay_actions_json:
        if task != "forage":
            raise ValueError("Recorded policy-action replay is currently supported only for Forage")
        replay_history = json.loads(Path(args.replay_actions_json).read_text(encoding="utf-8"))["history"]
        replay_actions = [entry["action"] for entry in replay_history]
    success = False
    joint_limit_violation_steps = 0
    strike_pinch_action = None
    strike_initial_tool_state = None
    forage_leaf_index = 1
    forage_stage = "approach"
    forage_clear_frames = 0
    forage_lost_contact_frames = 0
    scripted_joint_target = raw.hand.data.joint_pos.clone()
    branch_reference_joints = None
    if task == "branch" and args.branch_policy_demo:
        # Normalized legal-ROM targets from a physical opposition grasp.
        # Unknown joints retain the native reset target for evolved hands.
        fractions = {
            "link_1_thumb_spread_joint": .23510,
            "link_0_0_to_link_1_0": .74946, "link_0_0_to_link_2_0": .74300,
            "link_0_0_to_link_3_0": .74443, "link_0_0_to_link_4_0": .76872,
            "link_0_0_to_link_5_0": .73966,
            "link_1_0_to_link_1_1": .72845, "link_2_0_to_link_2_1": .72536,
            "link_3_0_to_link_3_1": .69317, "link_4_0_to_link_4_1": .61976,
            "link_5_0_to_link_5_1": .52340,
            "link_1_1_to_link_1_2": .76646, "link_2_1_to_link_2_2": .72801,
            "link_3_1_to_link_3_2": .70921, "link_4_1_to_link_4_2": .74595,
            "link_5_1_to_link_5_2": .74889,
        }
        branch_reference_joints = initial_joint_pos.clone()
        lower, upper = raw.hand_dof_lower_limits, raw.hand_dof_upper_limits
        for name, fraction in fractions.items():
            if name in raw.hand.joint_names:
                joint_id = raw.hand.joint_names.index(name)
                branch_reference_joints[:, joint_id] = lower[:, joint_id] + fraction * (upper[:, joint_id] - lower[:, joint_id])
    raw._compute_intermediate_values()
    initial_geometry = {
        "fingertips_world_m": raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
    }
    if task in ("branch", "strike"):
        task_object = raw.branch if task == "branch" else raw.cone
        initial_state = {
            "task": task,
            "joint_positions": dict(zip(raw.hand.joint_names, raw.hand.data.joint_pos[0].detach().cpu().tolist())),
            "hand_root_pose": raw.hand.data.root_state_w[0, :7].detach().cpu().tolist(),
            "object_root_pose": task_object.data.root_state_w[0, :7].detach().cpu().tolist(),
        }
        state_path = Path(args.metrics).with_suffix(".initial_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(initial_state, indent=2), encoding="utf-8")
    if adaptive_geometry is not None:
        initial_geometry["adaptive_pregasp"] = adaptive_geometry
    if task == "branch":
        initial_geometry["object_world_m"] = raw.branch.data.root_pos_w[0].detach().cpu().tolist()
    elif task == "forage":
        initial_geometry["food_world_m"] = raw.food.data.root_pos_w[0].detach().cpu().tolist()
    else:
        initial_geometry["tool_world_m"] = raw.cone.data.root_pos_w[0].detach().cpu().tolist()
        strike_initial_tool_state = raw.cone.data.root_state_w[:1].clone()
    try:
        max_steps = int(raw.max_episode_length) - 1 if task == "forage" else (320 if task == "strike" else 240)
        for step in range(max_steps):
            if task == "branch" and args.branch_policy_demo:
                alpha = min(1.0, max(0.0, (step - 10) / 130.0))
                desired = initial_joint_pos + alpha * (branch_reference_joints - initial_joint_pos)
                lower, upper = raw.hand_dof_lower_limits, raw.hand_dof_upper_limits
                action = torch.zeros((1, 20), device=raw.device)
                for finger in range(5):
                    joint_ids = raw.cartesian_ik.joint_ids[finger].tolist()
                    if finger > 0:
                        joint_ids = [index for index in joint_ids if "spread" not in raw.hand.joint_names[index]]
                    for channel, joint_id in enumerate(joint_ids[:4]):
                        action[:, 4 * finger + channel] = (
                            2.0 * (desired[:, joint_id] - lower[:, joint_id])
                            / (upper[:, joint_id] - lower[:, joint_id]).clamp_min(1.0e-6) - 1.0
                        ).clamp(-1.0, 1.0)
                raw.scripted_joint_target = None
            elif task == "branch":
                raw._compute_intermediate_values()
                approach = min(1.0, max(0.0, (step - 10) / 130.0))
                raw.scripted_joint_target = initial_joint_pos + approach * (closure_target - initial_joint_pos)
                action = torch.zeros((1, 20), device=raw.device)
                action[:, 15:20] = 0.95 * approach
            elif task == "forage" and replay_actions is not None:
                if step >= len(replay_actions):
                    break
                action = torch.tensor(replay_actions[step], device=raw.device, dtype=torch.float32).view(1, 23)
                forage_stage = "replay"
                selected_force = 0.0
            elif task == "forage":
                raw._compute_intermediate_values()
                leaf_asset = raw.leaf_two if forage_leaf_index == 1 else raw.leaf_one
                leaf = leaf_asset.data.root_pos_w.clone()
                away = leaf[:, :2] - raw.food.data.root_pos_w[:, :2]
                away /= torch.linalg.vector_norm(away, dim=-1, keepdim=True).clamp_min(1e-5)
                tip = raw.cartesian_ik.fingertip_positions_world()[:, 1 if forage_leaf_index == 1 else 3]
                sensor = raw.leaf_two_contact if forage_leaf_index == 1 else raw.leaf_one_contact
                finger_id = 2 if forage_leaf_index == 1 else 4
                filter_ids = [index for index, path in enumerate(sensor.cfg.filter_prim_paths_expr)
                              if path.rsplit("/", 1)[-1].startswith(f"link_{finger_id}_")]
                if not filter_ids:
                    raise RuntimeError(f"No contact channels for Forage pushing finger {finger_id}")
                selected_force = float(torch.linalg.vector_norm(
                    sensor.data.force_matrix_w[0, 0, filter_ids], dim=-1
                ).sum())
                contact = selected_force >= cfg.minimum_leaf_contact_force
                distance = float(raw._leaf_distances()[0, forage_leaf_index])
                forage_clear_frames = forage_clear_frames + 1 if distance >= cfg.leaf_clear_distance + 0.005 else 0
                if forage_clear_frames >= 3:
                    forage_stage = "lift"
                desired_tip = leaf.clone()
                direction3 = torch.cat((away, torch.zeros_like(away[:, :1])), dim=-1)
                direction_local = quat_apply(quat_conjugate(leaf_asset.data.root_quat_w), direction3)
                leaf_cfg = cfg.leaf_two_cfg if forage_leaf_index == 1 else cfg.leaf_one_cfg
                half_extents = torch.tensor(leaf_cfg.spawn.size, device=raw.device) / 2.0
                # Intersect the centre ray with the oriented box; support
                # projection gives a different point outside diagonal edges.
                ray_limits = torch.where(
                    direction_local.abs() > 1.0e-6,
                    half_extents / direction_local.abs().clamp_min(1.0e-6),
                    torch.full_like(direction_local, float("inf")),
                )
                edge_radius = ray_limits.min(dim=-1, keepdim=True).values
                desired_tip[:, :2] -= away * (edge_radius - 0.002)
                if forage_stage == "approach":
                    desired_tip[:, :2] -= away * 0.006
                    desired_tip[:, 2] += 0.025
                    if float(torch.linalg.vector_norm(tip - desired_tip)) < 0.005:
                        forage_stage = "contact"
                elif forage_stage == "contact":
                    if contact:
                        forage_stage = "sweep"
                        forage_sweep_direction = away.clone()
                        forage_lost_contact_frames = 0
                elif forage_stage == "sweep":
                    desired_tip[:, :2] = tip[:, :2] + forage_sweep_direction * 0.003
                    forage_lost_contact_frames = 0 if contact else forage_lost_contact_frames + 1
                    if forage_lost_contact_frames >= 12:
                        forage_stage = "approach"
                elif forage_stage == "lift":
                    desired_tip = tip.clone()
                    desired_tip[:, 2] = leaf[:, 2] + 0.025
                    if float(tip[0, 2] - leaf[0, 2]) >= 0.020 and forage_leaf_index == 1:
                        forage_leaf_index = 0
                        forage_stage = "approach"
                        forage_clear_frames = 0
                root = raw.hand.data.root_pos_w
                desired_root = root + torch.clamp(desired_tip - tip, -0.003, 0.003)
                neutral_root = raw.hand.data.default_root_state[:, :3] + raw.scene.env_origins
                action = torch.zeros((1, 23), device=raw.device)
                action[:, 20:23] = torch.clamp((desired_root - neutral_root) / raw.wrist_limits, -1.0, 1.0)
            else:
                # State machine: establish an opposing physical pinch first;
                # only after the environment's sustained-contact gate opens do
                # we translate the wrist to strike.  No tool pose is written.
                raw._compute_intermediate_values()
                grip_alpha = min(1.0, (step + 1) / 60.0)
                raw.scripted_joint_target = initial_joint_pos + grip_alpha * (
                    closure_target - initial_joint_pos
                )
                action = torch.zeros((1, 23), device=raw.device)
                if bool(raw.tool_was_held[0]):
                    height_error = raw.cone_tip_pos[:, 2] - raw.strike_target_pos[:, 2]
                    # Wrist actions are offsets from reset, not incremental motion.
                    neutral_root = raw.hand.data.default_root_state[:, :3] + raw.scene.env_origins
                    desired_root = raw.hand.data.root_pos_w.clone()
                    desired_root[:, 2] -= torch.clamp(height_error + 0.002, 0.0, 0.0005)
                    action[:, 20:23] = torch.clamp(
                        (desired_root - neutral_root) / raw.wrist_action_scale, -1.0, 1.0
                    )
            observation_before = raw._get_observations()["policy"][0].detach().cpu().numpy().astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            bc_observations.append(observation_before)
            bc_actions.append(action[0].detach().cpu().numpy().astype(np.float32))
            if task == "branch" and args.branch_policy_demo:
                joints = raw.hand.data.joint_pos[0]
                limits = raw.hand.root_physx_view.get_dof_limits()[0].to(joints.device)
                joint_limit_violation_steps += int(bool(((joints < limits[:, 0] - 0.005) | (joints > limits[:, 1] + 0.005)).any()))
            raw._compute_intermediate_values()
            entry = {"step": step, "reward": float(reward[0].item()),
                     "terminated": bool(terminated[0]), "truncated": bool(truncated[0]),
                     "state_after_auto_reset": bool(terminated[0] or truncated[0])}
            if task == "branch":
                forces = torch.norm(raw.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1)[0]
                entry.update(
                    thumb_force_n=float(forces[0]),
                    fingertip_forces_n=forces.detach().cpu().tolist(),
                    other_contact_count=int((forces[1:] >= cfg.branch_contact_force_threshold).sum()),
                    hold_steps=int(raw.branch_success_streak[0]),
                    fingertip_positions_m=raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
                    object_position_m=raw.branch.data.root_pos_w[0].detach().cpu().tolist(),
                    object_quaternion_wxyz=raw.branch.data.root_quat_w[0].detach().cpu().tolist(),
                )
                success = success or entry["hold_steps"] >= cfg.branch_success_hold_steps
            elif task == "forage":
                distances = raw._leaf_distances()[0]
                entry.update(
                    leaf_one_distance_m=float(distances[0]), leaf_two_distance_m=float(distances[1]),
                    leaf_hand_contact_forces_n=raw.leaf_hand_contact_forces[0].detach().cpu().tolist(),
                    leaf_contact_seen=raw.leaf_contact_seen[0].detach().cpu().tolist(),
                    invalid_leaf_motion=bool(raw.invalid_leaf_motion[0]),
                    hand_root_position_m=raw.hand.data.root_pos_w[0].detach().cpu().tolist(),
                    fingertip_positions_m=raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
                    leaf_one_position_m=raw.leaf_one.data.root_pos_w[0].detach().cpu().tolist(),
                    leaf_two_position_m=raw.leaf_two.data.root_pos_w[0].detach().cpu().tolist(),
                    action=action[0].detach().cpu().tolist(),
                    controller_stage=forage_stage,
                    target_leaf=forage_leaf_index,
                    pushing_finger_force_n=selected_force,
                )
                success = success or bool(raw.success_achieved[0]) or float(reward[0]) >= 999.0
            else:
                force = torch.norm(raw.strike_object_force[0]).item()
                distance = torch.norm(raw.cone_tip_pos[0, :2] - raw.strike_target_pos[0, :2]).item()
                finger_forces = raw.tool_fingertip_forces[0]
                entry.update(
                    impact_force_n=float(force),
                    tip_xy_error_m=float(distance),
                    tool_held=bool(raw.tool_was_held[0]),
                    thumb_tool_force_n=float(finger_forces[0]),
                    fingertip_forces_n=finger_forces.detach().cpu().tolist(),
                    long_tool_contact_count=int((finger_forces[1:] >= cfg.tool_finger_contact_force_threshold).sum()),
                    grasp_hold_steps=int(raw.tool_grasp_streak[0]),
                    fingertip_positions_m=raw.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
                    tool_position_m=raw.cone.data.root_pos_w[0].detach().cpu().tolist(),
                )
                success = success or entry["reward"] >= 1000.0
            history.append(entry)
            if args.preflight and task == "forage":
                progress_path = Path(args.metrics).with_suffix(".steps.jsonl")
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                with progress_path.open("a", encoding="utf-8") as progress_file:
                    progress_file.write(json.dumps(entry) + "\n")
                if step % 25 == 0:
                    print("[PREFLIGHT] Forage step", step, forage_stage,
                          entry["leaf_one_distance_m"], entry["leaf_two_distance_m"], flush=True)
            if writer is not None:
                writer.append_data(u8(env.render()))
            # Keep recordings useful even when an environment terminates early.
            if step + 1 >= args.min_video_steps and (success or bool(terminated[0]) or bool(truncated[0])):
                break
    finally:
        if writer is not None:
            writer.close()
    bc_path = Path(args.metrics).with_suffix('.trace.npz')
    if bc_observations:
        np.savez_compressed(
            bc_path,
            observations_before_step=np.asarray(bc_observations),
            submitted_actions=np.asarray(bc_actions),
            actions_control_fingers=np.full(len(bc_actions), task == "forage" or (task == "branch" and args.branch_policy_demo)),
            scene_unmodified=np.full(len(bc_actions), task == "forage" or args.training_scene),
        )
    if task == "branch" and args.branch_policy_demo and joint_limit_violation_steps:
        success = False
    summary = {"task": task, "morphology": args.individual_key or "human_hand", "success": success, "curriculum_stage": cfg.curriculum_stage, "reset_dof_pos_noise": cfg.reset_dof_pos_noise, "training_scene": bool(args.training_scene), "effective_drives": effective_drives, "steps_executed": len(history), "scripted_trace": str(bc_path), "joint_limit_violation_steps": joint_limit_violation_steps if task == "branch" and args.branch_policy_demo else None, "controller": "branch_joint_target_v1" if task == "branch" and args.branch_policy_demo else None, "initial_geometry": initial_geometry, "history": history}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    env.close()
    app.close()
    if args.preflight and not success:
        raise SystemExit(2)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in {"branch", "forage", "strike"} else None
    if task is None:
        raise SystemExit("Usage: scripted_adaptive_task_demo.py {branch|forage|strike} ...")
    main(task)
