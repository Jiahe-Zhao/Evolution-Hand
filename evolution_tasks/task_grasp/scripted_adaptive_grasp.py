"""Record a morphology-adaptive five-fingertip Grasp demonstration."""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record an adaptive five-finger Grasp demonstration.")
parser.add_argument("--lineage_json", help="Optional evolution result JSON containing lineage data.")
parser.add_argument("--individual_key", help="Individual key in lineage_json, for example 15_0.")
parser.add_argument("--output", required=True, help="Output MP4 path.")
parser.add_argument("--metrics", required=True, help="Output JSON trajectory and metrics path.")
parser.add_argument("--audit_mesh", action="store_true", help="Record source-mesh sphere clearance each replay step.")
parser.add_argument("--project_closure", action="store_true",
                    help="Diagnostic only: project the entire closure toward the first sphere contact.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--approach_steps", type=int, default=45)
parser.add_argument("--close_steps", type=int, default=90)
parser.add_argument("--hold_steps", type=int, default=20)
parser.add_argument(
    "--stabilize_steps", type=int, default=190,
    help="Keep the reset ball pose fixed during the scripted pre-grasp only.",
)
parser.add_argument(
    "--closure_fraction", type=float, default=0.12,
    help="Fraction of each morphology's safe flexion ROM used by the closure.",
)
parser.add_argument("--script_stiffness", type=float, default=None)
parser.add_argument("--script_damping", type=float, default=None)
parser.add_argument("--script_velocity_limit", type=float, default=None)
parser.add_argument("--script_effort_limit", type=float, default=None)
parser.add_argument("--force_threshold", type=float, default=0.10)
parser.add_argument("--preflight", action="store_true", help="Run physical success checks without rendering video.")
parser.add_argument("--contact_radius", type=float, default=0.021)
parser.add_argument(
    "--palm_residual",
    type=float,
    default=0.0,
    help="Optional IK residual along the palm vector. Zero preserves the geometric surface target.",
)
parser.add_argument(
    "--object_offset_world",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="Explicit world-frame offset from the task reset pose, recorded in metrics.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = not args.preflight
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_conjugate


NUM_FINGERS = 5
# Match the environment's Cartesian step scale. The target is fixed after reset
# so every finger closes along one stable inward ray instead of chasing a moving ball.
POSITION_SCALE = torch.tensor((0.005, 0.005, 0.005))


def _frame_u8(frame):
    if isinstance(frame, (tuple, list)):
        frame = frame[0]
    return np.asarray(frame).astype(np.uint8)


def _prepare_morphology(output_dir: Path):
    """Build one lineage morphology in an evaluation-local import path."""
    code_root = Path(os.environ.get("EVOLUTION_CODE_ROOT", "/home/zjh/Evolution_PC"))
    sys.path.insert(0, str(code_root / "Isaaclab_other"))
    from code_to_urdf import generate_urdf_from_dict
    from human_hand_agent import initial_agent_hand
    from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
    from mirror_agent import create_mirror_hand

    if bool(args.lineage_json) != bool(args.individual_key):
        raise ValueError("--lineage_json and --individual_key must be supplied together")
    if args.lineage_json:
        with Path(args.lineage_json).open(encoding="utf-8") as file:
            hand = json.load(file)["lineage"][args.individual_key]["urdf_info"]
    else:
        hand = dict(initial_agent_hand)

    morphology_key = (args.individual_key or "current_human_hand").replace("/", "_")
    morphology_dir = output_dir / "morphology" / morphology_key
    right_urdf = morphology_dir / "right" / "urdf" / "current_agent.urdf"
    left_urdf = morphology_dir / "left" / "urdf" / "current_agent.urdf"
    right_urdf.parent.mkdir(parents=True, exist_ok=True)
    left_urdf.parent.mkdir(parents=True, exist_ok=True)
    generate_urdf_from_dict(
        hand,
        output_dir=str(morphology_dir / "right" / "meshes"),
        output_urdf=str(right_urdf),
    )
    left_hand = create_mirror_hand(hand, f"{morphology_key}_adaptive_grasp_left")
    generate_urdf_from_dict(
        left_hand,
        output_dir=str(morphology_dir / "left" / "meshes"),
        output_urdf=str(left_urdf),
    )

    override_root = morphology_dir / "python_overrides" / "isaaclab_tasks" / "evolution_tasks"
    right_cfg = override_root / "current_right_hand" / "current_right_hand_cfg.py"
    left_cfg = override_root / "current_left_hand" / "current_left_hand_cfg.py"
    right_cfg.parent.mkdir(parents=True, exist_ok=True)
    left_cfg.parent.mkdir(parents=True, exist_ok=True)
    parse_urdf_and_generate_articulation_cfg(str(right_urdf), str(right_urdf), str(right_cfg))
    parse_urdf_and_generate_articulation_cfg(str(left_urdf), str(left_urdf), str(left_cfg))

    import isaaclab_tasks.evolution_tasks as evolution_tasks

    override_path = str(override_root)
    if override_path not in evolution_tasks.__path__:
        evolution_tasks.__path__.insert(0, override_path)
    importlib.invalidate_caches()

    # Match train_worker's explicit generated-right-hand binding, even though
    # the historical Grasp prim path is named LeftRobot.
    spec = importlib.util.spec_from_file_location("adaptive_grasp_hand_cfg", right_cfg)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated hand config: {right_cfg}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available_joints = set(module.MORPHOLOGY_CONTRACT["all_actuated_joints"])
    fingertip_names = []
    for finger_id in range(1, NUM_FINGERS + 1):
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
    return fingertip_names, module.CURRENT_HAND_CFG, available_joints


def _closure_targets(
    object_anchor: torch.Tensor,
    approach_directions: torch.Tensor,
    surface_radius: float,
) -> torch.Tensor:
    """Return fixed inward targets for all five fingers.

    The old controller recomputed a ray from the current tip to the current
    object position every step. Once the ball moved, that made the targets
    reverse direction. This version freezes the reset geometry and produces a
    coordinated five-finger closure.
    """
    object_pos = object_anchor
    return object_pos.unsqueeze(1) + approach_directions * surface_radius


def _adaptive_action(
    raw_env,
    tip_ids: list[int],
    phase: str,
    object_anchor: torch.Tensor,
    approach_directions: torch.Tensor,
) -> torch.Tensor:
    """Map fixed inward targets to the current 20D Cartesian IK action."""
    action = torch.zeros((1, 20), dtype=torch.float32, device=raw_env.device)
    tip_pos = raw_env.cartesian_ik.fingertip_positions_world()
    if phase == "approach":
        surface_radius = args.contact_radius + 0.010
    elif phase == "hold":
        surface_radius = max(0.001, args.contact_radius - 0.003)
    else:
        surface_radius = args.contact_radius
    target = _closure_targets(object_anchor, approach_directions, surface_radius)
    delta_world = target - tip_pos
    root_quat = raw_env.hand.data.root_quat_w
    local_quat = quat_conjugate(root_quat).unsqueeze(1).expand(-1, NUM_FINGERS, -1)
    delta_local = quat_apply(local_quat, delta_world)
    scale = POSITION_SCALE.to(raw_env.device).view(1, 1, 3)
    action[:, :15] = torch.clamp(delta_local / scale, -1.0, 1.0).reshape(1, 15)
    if phase != "approach":
        action[:, 15:] = args.palm_residual
    return action


def _old_surface_targets(raw_env, tip_ids: list[int], surface_radius: float) -> torch.Tensor:
    """Legacy helper retained for old recorded scripts."""
    object_pos = raw_env.grasp_object.data.root_pos_w[:, 0:3]
    tip_pos = raw_env.cartesian_ik.fingertip_positions_world()
    direction = tip_pos - object_pos.unsqueeze(1)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-5)
    return object_pos.unsqueeze(1) + direction * surface_radius


def _build_joint_closure_target(
    raw_env, closure_fraction: float, thumb_opposition: float = 0.0
) -> torch.Tensor:
    """Build one synchronized, limit-aware closure target for the loaded hand."""
    hard_limits = raw_env.hand.root_physx_view.get_dof_limits().to(raw_env.device)
    safe_limits = raw_env.cartesian_ik._safe_joint_limits(hard_limits)
    lower, upper = safe_limits[..., 0], safe_limits[..., 1]
    target = raw_env.hand.data.joint_pos.clone()
    for joint_index, joint_name in enumerate(raw_env.hand.joint_names):
        if joint_name == "link_1_thumb_spread_joint":
            target[:, joint_index] = torch.clamp(
                torch.full_like(target[:, joint_index], thumb_opposition),
                lower[:, joint_index],
                upper[:, joint_index],
            )
        elif "mcp_spread_joint" in joint_name:
            target[:, joint_index] = torch.zeros_like(target[:, joint_index])
        elif any(
            joint_name in {
                f"link_{finger_id}_0_to_link_{finger_id}_1",
                f"link_{finger_id}_1_to_link_{finger_id}_2",
            }
            for finger_id in range(1, 6)
        ) or any(
            joint_name == f"link_0_0_to_link_{finger_id}_0" for finger_id in range(1, 6)
        ):
            target[:, joint_index] = lower[:, joint_index] + closure_fraction * (
                upper[:, joint_index] - lower[:, joint_index]
            )
    target = raw_env.cartesian_ik._apply_flexion_coupling(target)
    return torch.clamp(target, lower, upper)


def _set_hand_pose(raw_env, target: torch.Tensor) -> None:
    raw_env.hand.write_joint_state_to_sim(target, torch.zeros_like(target))
    raw_env.hand.set_joint_position_target(target)
    raw_env.prev_targets[:] = target
    raw_env.cur_targets[:] = target
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env._compute_intermediate_values()


def _calibrate_physical_pregasp(env, raw_env, tip_ids: list[int], initial_joint_pos: torch.Tensor, mesh_auditor=None):
    """Find a legal morphology-specific grasp using measured contact forces."""
    zero_action = torch.zeros((1, 20), dtype=torch.float32, device=raw_env.device)
    offsets = tuple(itertools.product((-0.016, -0.008, 0.0, 0.008, 0.016), repeat=3))
    reset_support_position = raw_env.grasp_object.data.root_pos_w[:1].clone()
    geometric_candidates = []
    palm_workspace_candidates = []
    support_candidates = []
    reset_candidates = []
    best_failed_sustain = []
    best_failed_sustain_count = -1
    for fraction in (0.08, 0.20, 0.35, 0.50):
        for thumb_angle in (0.0, -0.70, -1.40, -2.10, -2.50):
            pregrasp_target = _build_joint_closure_target(raw_env, fraction, thumb_angle)
            compression_target = _build_joint_closure_target(
                raw_env, min(0.90, fraction + 0.36), thumb_angle - 0.25
            )
            # A useful grasp center lies on the palm side of the long-finger
            # pads.  Averaging an extreme thumb pose with long fingertips can
            # put the sphere beyond every finger's Cartesian workspace.
            _set_hand_pose(raw_env, pregrasp_target)
            pregrasp_tips = raw_env.cartesian_ik.fingertip_positions_world().clone()
            palm_center = raw_env.hand.data.root_pos_w[:, 0:3]
            support_point = raw_env._compute_proximal_support_point([0]).clone()
            for pair in itertools.combinations(range(1, NUM_FINGERS), 2):
                long_midpoint = pregrasp_tips[:, pair].mean(dim=1)
                palm_direction = palm_center - long_midpoint
                palm_direction = palm_direction / torch.linalg.vector_norm(
                    palm_direction, dim=-1, keepdim=True
                ).clamp_min(1e-5)
                base_position = long_midpoint + palm_direction * (args.contact_radius + 0.004)
                for offset_xyz in offsets:
                    position = base_position + torch.tensor(
                        offset_xyz, dtype=torch.float32, device=raw_env.device
                    ).view(1, 3)
                    selected_tips = pregrasp_tips[:, (0, *pair)]
                    radii = torch.linalg.vector_norm(
                        selected_tips - position.unsqueeze(1), dim=-1
                    )
                    desired_center_distance = args.contact_radius + 0.004
                    geometric_error = float(
                        torch.abs(radii - desired_center_distance).mean()
                    )
                    palm_workspace_candidates.append(
                        (geometric_error, pregrasp_target.clone(), compression_target.clone(),
                         position.clone(), pair)
                    )
                    # Keep a second family of candidates on the actual
                    # first-phalanx support shelf.  A fingertip-only centroid
                    # can be geometrically reachable but physically floating.
                    support_position = support_point + torch.tensor(
                        offset_xyz, dtype=torch.float32, device=raw_env.device
                    ).view(1, 3)
                    support_radii = torch.linalg.vector_norm(
                        pregrasp_tips[:, (0, *pair)] - support_position.unsqueeze(1), dim=-1
                    )
                    support_error = float(
                        torch.abs(support_radii - (args.contact_radius + 0.004)).mean()
                    )
                    support_candidates.append(
                        (support_error, pregrasp_target.clone(), compression_target.clone(),
                         support_position.clone(), pair)
                    )
                    reset_position = reset_support_position + torch.tensor(
                        offset_xyz, dtype=torch.float32, device=raw_env.device
                    ).view(1, 3)
                    reset_radii = torch.linalg.vector_norm(
                        pregrasp_tips[:, (0, *pair)] - reset_position.unsqueeze(1), dim=-1
                    )
                    reset_error = float(
                        torch.abs(reset_radii - (args.contact_radius + 0.004)).mean()
                    )
                    reset_candidates.append(
                        (reset_error, pregrasp_target.clone(), compression_target.clone(),
                         reset_position.clone(), pair)
                    )
            # Place the ball in the envelope produced by the commanded
            # closure, not by the open reset pose.
            _set_hand_pose(raw_env, compression_target)
            tips = raw_env.cartesian_ik.fingertip_positions_world().clone()
            for pair in itertools.combinations(range(1, NUM_FINGERS), 2):
                center = tips[:, (0, *pair)].mean(dim=1)
                for offset_xyz in offsets:
                    position = center + torch.tensor(
                        offset_xyz, dtype=torch.float32, device=raw_env.device
                    ).view(1, 3)
                    selected_tips = tips[:, (0, *pair)]
                    radii = torch.linalg.vector_norm(selected_tips - position.unsqueeze(1), dim=-1)
                    desired_center_distance = args.contact_radius + 0.004
                    geometric_error = float(
                        torch.abs(radii.mean() - desired_center_distance) + radii.std(unbiased=False)
                    )
                    geometric_candidates.append(
                        (geometric_error, pregrasp_target.clone(), compression_target.clone(),
                         position.clone(), pair)
                    )

    # Only the geometrically closest candidates need a PhysX probe. The
    # physical validation below remains the authoritative success gate.
    geometric_candidates.sort(key=lambda item: item[0])
    palm_workspace_candidates.sort(key=lambda item: item[0])
    support_candidates.sort(key=lambda item: item[0])
    reset_candidates.sort(key=lambda item: item[0])
    # Preflight is executed for every evolved morphology. Eight diverse,
    # geometry-ranked probes are enough to reject unreachable hands without
    # turning the gate itself into a second training loop.
    geometric_candidates = (
        geometric_candidates[:8]
        + palm_workspace_candidates[:8]
        + support_candidates[:8]
        + reset_candidates[:8]
    )
    best = None
    for _, pregrasp_target, compression_target, position, pair in geometric_candidates:
        # Clear cached impulses from the previous contact probe.
        env.reset(seed=args.seed)
        raw_env.scripted_joint_target = None
        state = raw_env.grasp_object.data.default_root_state[:1].clone()
        state[:, 0:3] = position
        state[:, 7:13] = 0.0
        raw_env.episode_length_buf.zero_()
        raw_env.reset_buf.zero_()
        raw_env.milestone_streaks.zero_()
        raw_env.milestone_claimed.zero_()
        raw_env.in_hand_pos[:1] = position
        _set_hand_pose(raw_env, pregrasp_target)
        if mesh_auditor is not None:
            clearance = mesh_auditor.measure(
                position[0].detach().cpu().numpy(),
                float(raw_env.cfg.grasp_object_cfg.spawn.radius),
            )
            # A candidate whose sphere already penetrates a collision solid is
            # not a grasp candidate; it is a task-placement artifact.
            if float(clearance["clearance_m"]) < -1e-4:
                continue
        _set_hand_pose(raw_env, pregrasp_target)
        # The old five-step probe could not reach ``compression_target`` at
        # the configured 0.8 rad/s velocity limit.  Search the full legal
        # closure ramp and retain the first physically supported envelope.
        probe_best = None
        for probe_step in range(48):
            alpha = (probe_step + 1) / 48.0
            raw_env.scripted_joint_target = pregrasp_target + alpha * (
                compression_target - pregrasp_target
            )
            raw_env.grasp_object.write_root_state_to_sim(state)
            env.step(zero_action)
            raw_env._compute_intermediate_values()
            forces = raw_env.full_hand_contact_forces[0]
            second_long = torch.topk(forces[1:], k=2).values[-1]
            score = float(torch.minimum(forces[0], second_long))
            current_tips = raw_env.cartesian_ik.fingertip_positions_world()
            thumb_vector = current_tips[:, 0] - position
            long_vector = current_tips[:, pair].mean(dim=1) - position
            opposition = -torch.nn.functional.cosine_similarity(
                thumb_vector, long_vector, dim=-1
            )
            if float(opposition[0]) < 0.20:
                score = 0.0
            peak = float(torch.max(forces))
            score -= max(0.0, peak - 10.0) * 0.05
            if probe_best is None or score > probe_best[0]:
                probe_best = (
                    score,
                    raw_env.prev_targets.clone(),
                    forces.detach().cpu().tolist(),
                )
        # Finish with the same morphology-aware Cartesian controller exposed
        # to PPO.  The scripted joint ramp only enters the local basin; IK
        # aligns each evolved fingertip with the actual sphere surface.
        if probe_best is not None and probe_best[0] < args.force_threshold:
            raw_env.scripted_joint_target = None
            probe_anchor = position
            probe_directions = raw_env.cartesian_ik.fingertip_positions_world() - probe_anchor.unsqueeze(1)
            probe_directions = probe_directions / torch.linalg.vector_norm(
                probe_directions, dim=-1, keepdim=True
            ).clamp_min(1e-5)
            for _ in range(35):
                raw_env.grasp_object.write_root_state_to_sim(state)
                action = _adaptive_action(
                    raw_env, tip_ids, "hold", probe_anchor, probe_directions
                )
                env.step(action)
                raw_env._compute_intermediate_values()
                forces = raw_env.full_hand_contact_forces[0]
                second_long = torch.topk(forces[1:], k=2).values[-1]
                score = float(torch.minimum(forces[0], second_long))
                current_tips = raw_env.cartesian_ik.fingertip_positions_world()
                thumb_vector = current_tips[:, 0] - position
                long_vector = current_tips[:, pair].mean(dim=1) - position
                opposition = -torch.nn.functional.cosine_similarity(
                    thumb_vector, long_vector, dim=-1
                )
                if float(opposition[0]) < 0.20:
                    score = 0.0
                peak = float(torch.max(forces))
                score -= max(0.0, peak - 10.0) * 0.05
                if score > probe_best[0]:
                    probe_best = (
                        score,
                        raw_env.prev_targets.clone(),
                        forces.detach().cpu().tolist(),
                    )
        if probe_best is None:
            continue
        score, hold_target, measured_forces = probe_best
        physical_pair = tuple(
            sorted(
                int(index) + 1
                for index in torch.topk(
                    torch.tensor(measured_forces[1:]), k=2
                ).indices.tolist()
            )
        )
        if best is None or score > best[0]:
            best = (score, hold_target, position.clone(), physical_pair, measured_forces)
        if score >= args.force_threshold:
            # Re-establish the best measured envelope before releasing the
            # sphere.  The search may have continued past that pose, and
            # releasing from the final probe state creates an artificial slip.
            hard_limits = raw_env.hand.root_physx_view.get_dof_limits().to(raw_env.device)
            preload_target = hold_target.clone()
            selected_fingers = (1, physical_pair[0] + 1, physical_pair[1] + 1)
            for joint_index, joint_name in enumerate(raw_env.hand.joint_names):
                if "spread_joint" in joint_name:
                    continue
                if any(
                    joint_name.startswith(f"link_{finger_id}_")
                    or f"to_link_{finger_id}_" in joint_name
                    for finger_id in selected_fingers
                ):
                    preload_target[:, joint_index] += 0.08 * (
                        hold_target[:, joint_index] - pregrasp_target[:, joint_index]
                    )
            preload_target = torch.clamp(
                preload_target, hard_limits[..., 0], hard_limits[..., 1]
            )
            _set_hand_pose(raw_env, preload_target)
            raw_env.scripted_joint_target = preload_target
            for _ in range(5):
                raw_env.grasp_object.write_root_state_to_sim(state)
                env.step(zero_action)
                raw_env._compute_intermediate_values()
            raw_env.episode_length_buf.zero_()
            raw_env.reset_buf.zero_()
            raw_env.milestone_streaks.zero_()
            raw_env.milestone_claimed.zero_()
            # This target was reached through the same morphology-aware IK
            # during the probe. Holding it here tests real free-object force
            # closure without injecting or pinning the sphere pose.
            raw_env.scripted_joint_target = preload_target
            sustained = False
            sustained_streak = 0
            sustained_forces = []
            for _ in range(30):
                _, reward, terminated, truncated, _ = env.step(zero_action)
                raw_env._compute_intermediate_values()
                active = raw_env.full_hand_contact_forces[0] >= raw_env.cfg.m3_contact_force_threshold
                physically_valid = bool(active[0] and active[1:].sum() >= 2)
                sustained_streak = sustained_streak + 1 if physically_valid else 0
                # Preflight validates the physical M3 definition directly.
                # Reward emission additionally depends on M1/M2 bookkeeping,
                # which is a curriculum concern rather than morphology
                # solvability.
                successful_terminal = physically_valid
                if bool(terminated[0] or truncated[0]) and not successful_terminal:
                    sustained_streak = 0
                sustained_forces.append(raw_env.full_hand_contact_forces[0].detach().cpu().tolist())
                if sustained_streak >= int(raw_env.cfg.m3_hold_steps):
                    sustained = True
                    break
            if sustained:
                _set_hand_pose(raw_env, initial_joint_pos)
                raw_env.scripted_joint_target = None
                return (score, preload_target.clone(), position.clone(), physical_pair,
                        measured_forces, True, sustained_forces)
            valid_steps = sum(
                forces[0] >= raw_env.cfg.m3_contact_force_threshold
                and sum(
                    force >= raw_env.cfg.m3_contact_force_threshold
                    for force in forces[1:]
                ) >= 2
                for forces in sustained_forces
            )
            if valid_steps > best_failed_sustain_count:
                best_failed_sustain_count = valid_steps
                best_failed_sustain = sustained_forces
    _set_hand_pose(raw_env, initial_joint_pos)
    raw_env.scripted_joint_target = None
    if best is None:
        raise RuntimeError("No physical Grasp pre-grasp candidate was evaluated.")
    return (*best, False, best_failed_sustain)


def _build_mcp_closure_target(raw_env, initial_joint_pos: torch.Tensor, full_target: torch.Tensor) -> torch.Tensor:
    """Close opposition and proximal flexion before folding distal joints."""
    target = initial_joint_pos.clone()
    for joint_index, joint_name in enumerate(raw_env.hand.joint_names):
        if joint_name == "link_1_thumb_spread_joint":
            target[:, joint_index] = full_target[:, joint_index]
        elif any(
            joint_name == f"link_0_0_to_link_{finger_id}_0" for finger_id in range(1, 6)
        ):
            target[:, joint_index] = full_target[:, joint_index]
    return target


def main() -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tips, generated_hand_cfg, available_joints = _prepare_morphology(output_path.parent)

    importlib.import_module("isaaclab_tasks.evolution_tasks.task_grasp")
    cfg_module = importlib.import_module(
        "isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg"
    )
    cfg = cfg_module.EvolutionGraspEnvCfg()
    initial_state = cfg.robot_cfg.init_state
    initial_state = initial_state.replace(joint_pos={
        name: value for name, value in (initial_state.joint_pos or {}).items()
        if name == ".*" or name in available_joints
    })
    cfg.robot_cfg = generated_hand_cfg.replace(prim_path=cfg.robot_cfg.prim_path).replace(init_state=initial_state)
    print("[PREFLIGHT] Morphology asset", cfg.robot_cfg.spawn.asset_path, flush=True)
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.seed = args.seed
    cfg.reset_dof_pos_noise = 0.0
    cfg.fingertip_body_names = tips
    cfg.contact_sensor_cfg.filter_prim_paths_expr = [
        f"/World/envs/env_.*/LeftRobot/{name}" for name in tips
    ]
    cfg.viewer.eye = (0.30, -0.30, 0.48)
    cfg.viewer.lookat = (0.0, 0.0, 0.34)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0

    overrides = {
        "stiffness": args.script_stiffness, "damping": args.script_damping,
        "velocity_limit_sim": args.script_velocity_limit,
        "effort_limit_sim": args.script_effort_limit,
    }
    if args.preflight and any(value is not None for value in overrides.values()):
        raise ValueError("Preflight must use training drives; remove script drive overrides")
    for actuator in cfg.robot_cfg.actuators.values():
        for key, value in overrides.items():
            if value is not None:
                setattr(actuator, key, {".*": value})
    effective_drives = {name: {key: getattr(actuator, key) for key in overrides}
                        for name, actuator in cfg.robot_cfg.actuators.items()}
    print("[PREFLIGHT] Effective drives", effective_drives, flush=True)

    env = gym.make(
        "Isaac-EvolutionHand-Grasp-v0", cfg=cfg,
        render_mode=None if args.preflight else "rgb_array",
    )
    raw_env = env.unwrapped
    env.reset(seed=args.seed)
    raw_env._compute_intermediate_values()
    if os.environ.get("EVOLUTION_DEBUG_JOINT_ORDER") == "1":
        hard_limits = raw_env.hand.root_physx_view.get_dof_limits()[0].detach().cpu().tolist()
        joint_positions = raw_env.hand.data.joint_pos[0].detach().cpu().tolist()
        print(
            "[DEBUG] runtime_joint_state="
            + json.dumps(list(zip(raw_env.hand.joint_names, joint_positions, hard_limits))),
            flush=True,
        )
    tip_ids = [raw_env.hand.body_names.index(name) for name in tips]
    initial_joint_pos = raw_env.hand.data.joint_pos.clone()
    # The scripted trajectory must use the same source-solid boundary as the
    # diagnostic. Otherwise a force-based contact can be reported after the
    # target has already driven the ball through a phalanx.
    from isaaclab_tasks.evolution_tasks.sphere_mesh_audit import SphereMeshAudit
    mesh_auditor = SphereMeshAudit(raw_env.hand)
    calibration = _calibrate_physical_pregasp(env, raw_env, tip_ids, initial_joint_pos, mesh_auditor)
    (
        calibration_score,
        closure_target,
        adaptive_object_position,
        calibration_pair,
        calibration_forces,
        calibration_sustained_success,
        calibration_sustained_forces,
    ) = calibration
    # Recenter the object on the actual closed fingertip envelope for this
    # morphology. The proximal support point is useful as a safe reset pose,
    # but it can be several centimetres away from the evolved fingertips after
    # closure. Using the calibrated thumb + two long-finger tips keeps the
    # script morphology-adaptive instead of hard-coding a hand-specific offset.
    _set_hand_pose(raw_env, closure_target)
    raw_env.hand.write_joint_state_to_sim(closure_target, torch.zeros_like(closure_target))
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env._compute_intermediate_values()
    closed_tip_positions = raw_env.cartesian_ik.fingertip_positions_world()
    selected_tip_indices = (0, int(calibration_pair[0]), int(calibration_pair[1]))
    adaptive_object_position = closed_tip_positions[:, selected_tip_indices].mean(dim=1)
    # The generated palm frame places the fingertip envelope distal and dorsal
    # to this raw mean; shift the sphere into the physical pad envelope.
    adaptive_object_position = adaptive_object_position + torch.tensor([0.030, 0.000, 0.020], device=raw_env.device)
    _set_hand_pose(raw_env, initial_joint_pos)
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env._compute_intermediate_values()
    # Reuse the current scene. Recreating a second DirectRLEnv inside one
    # SimulationApp can hang during PhysX teardown on headless machines.
    env.reset(seed=args.seed)
    raw_env.scripted_joint_target = None
    raw_env._compute_intermediate_values()
    tip_ids = [raw_env.hand.body_names.index(name) for name in tips]
    initial_joint_pos = raw_env.hand.data.joint_pos.clone()
    object_state = raw_env.grasp_object.data.default_root_state[:1].clone()
    object_state[:, 0:3] = adaptive_object_position
    object_state[:, 7:13] = 0.0
    raw_env.grasp_object.write_root_state_to_sim(object_state)
    raw_env.in_hand_pos[:1] = object_state[:, 0:3]
    _set_hand_pose(raw_env, initial_joint_pos)
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env._compute_intermediate_values()
    object_anchor = raw_env.grasp_object.data.root_pos_w[:, 0:3].clone()
    initial_tip_pos = raw_env.cartesian_ik.fingertip_positions_world().clone()
    approach_directions = initial_tip_pos - object_anchor.unsqueeze(1)
    approach_directions = approach_directions / torch.linalg.vector_norm(
        approach_directions, dim=-1, keepdim=True
    ).clamp_min(1e-5)
    offset = torch.tensor(args.object_offset_world, dtype=torch.float32, device=raw_env.device)
    if bool(torch.any(offset)):
        object_state = raw_env.grasp_object.data.default_root_state[:1].clone()
        object_state[:, 0:3] = raw_env.grasp_object.data.root_pos_w[:1] + offset
        # Keep the reset quaternion; only clear linear and angular velocity.
        object_state[:, 7:13] = 0.0
        raw_env.grasp_object.write_root_state_to_sim(object_state)
        raw_env.in_hand_pos[:1] = object_state[:, 0:3]
        raw_env.sim.forward()
        raw_env.scene.update(dt=0.0)
        raw_env._compute_intermediate_values()
        object_anchor = raw_env.grasp_object.data.root_pos_w[:, 0:3].clone()
        approach_directions = raw_env.cartesian_ik.fingertip_positions_world() - object_anchor.unsqueeze(1)
        approach_directions = approach_directions / torch.linalg.vector_norm(
            approach_directions, dim=-1, keepdim=True
        ).clamp_min(1e-5)
    stabilized_object_state = raw_env.grasp_object.data.root_state_w[:1].clone()
    stabilized_object_state[:, 7:13] = 0.0
    projection_report = {"enabled": bool(args.project_closure), "original_target": closure_target[0].detach().cpu().tolist()}
    if args.project_closure:
        start_target = initial_joint_pos.clone()
        safe_target = start_target.clone()
        last_clearance = None
        # Project along the morphology-specific closure ray.  The line search
        # is performed against the actual generated source meshes while the
        # ball remains at the calibrated position, so this cannot reward
        # penetration as a grasp.
        for index in range(1, 41):
            alpha = index / 40.0
            candidate = start_target + alpha * (closure_target - start_target)
            _set_hand_pose(raw_env, candidate)
            raw_env.grasp_object.write_root_state_to_sim(stabilized_object_state)
            raw_env.sim.forward()
            raw_env.scene.update(dt=0.0)
            raw_env._compute_intermediate_values()
            clearance = mesh_auditor.measure(
                raw_env.grasp_object.data.root_pos_w[0].detach().cpu().numpy(),
                float(raw_env.cfg.grasp_object_cfg.spawn.radius),
            )
            last_clearance = clearance
            if float(clearance["clearance_m"]) < -1.0e-4:
                break
            safe_target = candidate.clone()
        closure_target = safe_target
        projection_report.update({
            "projected_target": closure_target[0].detach().cpu().tolist(),
            "last_clearance": last_clearance,
            "projected_fraction": float(index - 1) / 40.0 if last_clearance and float(last_clearance["clearance_m"]) < -1.0e-4 else 1.0,
        })
        _set_hand_pose(raw_env, initial_joint_pos)
        raw_env.grasp_object.write_root_state_to_sim(stabilized_object_state)
        raw_env.sim.forward()
        raw_env.scene.update(dt=0.0)
        raw_env._compute_intermediate_values()
    if not args.preflight:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat, camera_prim_path="/OmniverseKit_Persp")
    writer = None if args.preflight else imageio.get_writer(output_path, fps=30, codec="libx264", quality=8)
    history: list[dict] = []
    all_five_streak = 0
    env_m3_success = False
    five_finger_success = False
    limit_violation_count = 0
    penetration_count = 0
    try:
        approach_steps = 30 if args.preflight else args.approach_steps
        close_steps = 90 if args.preflight else args.close_steps
        hold_steps = max(args.hold_steps, int(raw_env.cfg.m1_hold_steps + raw_env.cfg.m2_hold_steps + raw_env.cfg.m3_hold_steps) + 10)
        release_step = min(args.stabilize_steps, approach_steps + close_steps)
        total_steps = approach_steps + close_steps + hold_steps
        for step in range(total_steps):
            if step < approach_steps:
                phase = "approach"
            elif step < approach_steps + close_steps:
                phase = "close"
            else:
                phase = "hold"
            if step < approach_steps:
                alpha = 0.0
                target_joint_pos = initial_joint_pos
            elif step < approach_steps + close_steps:
                alpha = (step - approach_steps + 1) / max(close_steps, 1)
                target_joint_pos = initial_joint_pos + alpha * (closure_target - initial_joint_pos)
            else:
                alpha = 1.0
                target_joint_pos = closure_target
            # Replay the calibrated morphology-specific joint trajectory.
            # This validates physical grasping, not PPO's ability to discover it.
            raw_env.scripted_joint_target = target_joint_pos
            if step < release_step:
                raw_env.grasp_object.write_root_state_to_sim(stabilized_object_state)
                raw_env.sim.forward()
                raw_env.scene.update(dt=0.0)
            if step == release_step:
                raw_env.milestone_streaks.zero_()
                raw_env.milestone_claimed.zero_()
                raw_env.success_streaks.zero_()
                raw_env.successes.zero_()
                all_five_streak = 0
            action = torch.zeros((1, 20), device=raw_env.device)
            _, reward, terminated, truncated, _ = env.step(action)
            forces = raw_env.full_hand_contact_forces[0].detach().cpu().tolist()
            mesh_result = None
            if mesh_auditor is not None:
                mesh_result = mesh_auditor.measure(
                    raw_env.grasp_object.data.root_pos_w[0].detach().cpu().numpy(),
                    float(raw_env.cfg.grasp_object_cfg.spawn.radius),
                )
                penetration_count += int(float(mesh_result["clearance_m"]) < -1e-4)
            contact_flags = [float(force) >= args.force_threshold for force in forces]
            all_five_streak = all_five_streak + 1 if step >= release_step and all(contact_flags) else 0
            env_m3_success = env_m3_success or (step >= release_step and bool(raw_env.milestone_claimed[0, 2]))
            five_finger_success = five_finger_success or all_five_streak >= hold_steps
            hard_limits = raw_env.hand.root_physx_view.get_dof_limits()[0].to(raw_env.device)
            joint_pos = raw_env.hand.data.joint_pos[0]
            step_limit_violation = bool(
                (~torch.isfinite(joint_pos)).any()
                or (joint_pos < hard_limits[:, 0] - 0.005).any()
                or (joint_pos > hard_limits[:, 1] + 0.005).any()
            )
            limit_violation_count += int(step_limit_violation)
            history.append(
                {
                    "step": step,
                    "phase": phase,
                    "object_pinned": step < release_step,
                    "source_mesh_clearance": mesh_result,
                    "state_after_auto_reset": bool(terminated[0] or truncated[0]),
                    "joint_target_rad": target_joint_pos[0].detach().cpu().tolist(),
                    "action": action[0].detach().cpu().tolist(),
                    "tip_positions_world_m": raw_env.cartesian_ik.fingertip_positions_world()[0].detach().cpu().tolist(),
                    "object_position_world_m": raw_env.grasp_object.data.root_pos_w[0].detach().cpu().tolist(),
                    "object_anchor_world_m": object_anchor[0].detach().cpu().tolist(),
                    "joint_positions_rad": raw_env.hand.data.joint_pos[0].detach().cpu().tolist(),
                    "joint_limits_rad": raw_env.hand.root_physx_view.get_dof_limits()[0].detach().cpu().tolist(),
                    "joint_limit_violation": step_limit_violation,
                    "fingertip_contact_forces_n": forces,
                    "contact_flags": contact_flags,
                    "all_five_contact_streak": all_five_streak,
                    "environment_m3_success": env_m3_success,
                    "reward": float(reward[0]),
                }
            )
            if writer is not None:
                writer.append_data(_frame_u8(env.render()))
            if env_m3_success or penetration_count or bool(terminated[0] or truncated[0]):
                break
    finally:
        if writer is not None:
            writer.close()
        env.close()

    summary = {
        "effective_drives": effective_drives,
        "task": "Grasp",
        "success": bool(env_m3_success and limit_violation_count == 0 and penetration_count == 0),
        "source_mesh_penetration_steps": penetration_count,
        "source_mesh_penetration_tolerance_m": 1e-4,
        "controller": "morphology_adaptive_synchronized_joint_closure",
        "lineage_json": args.lineage_json,
        "individual_key": args.individual_key or "current_human_hand",
        "fingertip_body_names": tips,
        "object_offset_world_m": list(args.object_offset_world),
        "calibration_score": calibration_score,
        "calibration_long_finger_pair": list(calibration_pair),
        "calibration_contact_forces_n": calibration_forces,
        "calibration_sustained_success": calibration_sustained_success,
        "calibration_sustained_forces_n": calibration_sustained_forces,
        "success_definition": (
            f"thumb + at least 2 long fingertips >= {args.force_threshold} N for "
            f"{raw_env.cfg.m3_hold_steps} consecutive control steps"
        ),
        "environment_m3_success": env_m3_success,
        "five_finger_success": five_finger_success,
        "joint_limit_violation_steps": limit_violation_count,
        "steps_executed": len(history),
        "max_all_five_contact_streak": max((item["all_five_contact_streak"] for item in history), default=0),
        "max_fingertip_contact_forces_n": [
            max((item["fingertip_contact_forces_n"][index] for item in history), default=0.0)
            for index in range(NUM_FINGERS)
        ],
        "collision_mesh_audit": True,
        "collision_aware_target_projection": projection_report,
        "trajectory": history,
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "trajectory"}, indent=2))
    app.close()
    if args.preflight and not summary["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
