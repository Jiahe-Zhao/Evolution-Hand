"""Reproducible, per-episode evaluation for Evolution RL-Games policies."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from task_registry import TASKS


parser = argparse.ArgumentParser(description="Evaluate one fixed morphology and policy over fixed seeds.")
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--checkpoint", default="auto")
parser.add_argument("--curriculum_stage", choices=("auto", "stage1", "stage2"), default="auto")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--episodes", type=int, default=1, help="One process evaluates one episode; use run_reproducible_evaluation.sh for N episodes.")
parser.add_argument("--seed", type=int, default=7, help="First deterministic episode seed.")
parser.add_argument("--replay_until_step", type=int, default=0, help="Diagnostic: use saved actions through this step, then run the checkpoint policy.")
parser.add_argument("--replay_policy_trace", help="Replay a saved successful policy action trace with identical seed and scene.")
parser.add_argument("--audit_physics", action="store_true", help="Audit physical joint limits and Grasp source-mesh clearance.")
parser.add_argument("--export_success_bc", action="store_true", help="Save policy observations/actions only when the full task succeeds.")
parser.add_argument("--export_rollout_debug", action="store_true", help="Save policy observations/actions for diagnostics even on failure; never label them as BC.")
parser.add_argument("--strike_wrist_teacher", action="store_true", help="Probe a policy-action wrist teacher after Strike grasp; report as demonstration, not policy evaluation.")
parser.add_argument("--strike_wrist_step", type=float, default=0.05)
parser.add_argument("--strike_keep_policy_fingers", action="store_true")
parser.add_argument("--strike_fixed_wrist_actions", help="Three comma-separated normalized wrist targets for a slow Strike teacher probe.")
parser.add_argument("--strike_force_closure", type=float, help="Override five policy closure channels after grasp for a Strike teacher probe.")
parser.add_argument("--episode_index", type=int, default=0, help="Stable index assigned by the N-episode runner.")
parser.add_argument("--max_steps", type=int, default=0, help="0 uses the task's episode limit.")
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument(
    "--record_video",
    action="store_true",
    help="Render and retain episode video. Disabled by default for fast score-only evaluation.",
)
parser.add_argument(
    "--keep_failure_videos",
    action="store_true",
    help="Retain failed episode videos for physical-error auditing.",
)
parser.add_argument("--lineage_json", help="Optional lineage JSON used to rebuild a fixed evolved morphology.")
parser.add_argument("--individual_key", help="Lineage key such as '5_16'; required with --lineage_json.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = args.record_video
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import torch
from isaacsim.core.utils.viewports import set_camera_view
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


CAMERA_VIEWS = {
    "grasp": ((0.42, -0.42, 0.58), (0.0, 0.0, 0.26)),
    "branch": ((-0.34, -0.46, 0.58), (0.0, 0.0, 0.29)),
    "forage": ((0.36, -0.36, 0.42), (0.0, 0.0, 0.11)),
    "strike": ((-0.55, -0.50, 0.60), (-0.05, 0.01, 0.23)),
}


@dataclass
class MorphologyContext:
    override_task_root: Path
    body_names: set[str]
    morphology_contract: dict[str, Any]


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().flatten()[0].cpu())
    return float(value)


def _resolve_checkpoint() -> str:
    if args.checkpoint != "auto":
        if args.curriculum_stage == "auto":
            raise ValueError("Manual checkpoints require --curriculum_stage stage1 or stage2")
        return args.checkpoint
    if not args.lineage_json or not args.individual_key:
        raise ValueError("--checkpoint auto requires --lineage_json and --individual_key")
    with Path(args.lineage_json).open(encoding="utf-8") as file:
        individual = json.load(file)["lineage"][args.individual_key]
    metadata = individual.get("metadata", {})
    stage = metadata.get("selected_curriculum_stage", "stage1")
    if args.curriculum_stage != "auto" and args.curriculum_stage != stage:
        raise ValueError(f"Checkpoint lineage stage is {stage}, but --curriculum_stage={args.curriculum_stage}")
    run_names = metadata.get(f"{stage}_run_names", {})
    env_id = TASKS[args.task][0]
    run_name = next((value for key, value in run_names.items() if key.endswith(f":{env_id}")), None)
    if run_name is None:
        raise FileNotFoundError(f"No {stage} run metadata for {args.task}")
    root = Path(os.environ.get("EVOLUTION_CODE_ROOT", "/home/zjh/Evolution_PC"))
    nn_dir = root / "evolution_tasks" / "logs" / "evolution_task" / run_name / "nn"
    named = nn_dir / "evolution_task.pth"
    if named.is_file():
        return str(named)
    candidates = sorted(nn_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {nn_dir}")
    return str(candidates[0])


def _configure_checkpoint_controller(checkpoint: str) -> str | None:
    """Restore saved action and reset semantics before constructing an environment."""
    if args.task not in {"branch", "strike"}:
        return None
    path = Path(checkpoint).parent.parent / "params" / "policy_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"{args.task} checkpoint has no policy contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("task") != TASKS[args.task][0]:
        raise ValueError(f"{args.task} checkpoint task mismatch: {path}")
    controller = contract.get("controller")
    modes = ({"branch_joint_target_v1": "1", "cartesian_5finger_ik_v1": "0"}
             if args.task == "branch" else
             {"strike_joint_target_v1": "1", "cartesian_5finger_ik_v1": "0"})
    if controller not in modes:
        raise ValueError(f"Unknown {args.task} checkpoint controller: {controller}")
    os.environ["EVOLUTION_BRANCH_BC_MODE" if args.task == "branch" else "EVOLUTION_STRIKE_BC_MODE"] = modes[controller]
    if args.task == "strike" and controller == "strike_joint_target_v1":
        environment = contract.get("environment")
        required = {"EVOLUTION_STRIKE_RESET_THUMB_SPREAD", "EVOLUTION_STRIKE_PREGRASP_ACTION", "EVOLUTION_STRIKE_RESET_OFFSET_WORLD"}
        if not isinstance(environment, dict) or not required.issubset(environment):
            raise ValueError(f"Strike checkpoint lacks reset contract: {path}")
        for name in required:
            os.environ[name] = str(environment[name])
        teacher = contract.get("frozen_bc_teacher")
        if teacher:
            import hashlib
            teacher_path = Path(teacher["checkpoint"])
            if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != teacher["sha256"]:
                raise ValueError("Strike frozen BC teacher checkpoint hash mismatch")
            if teacher.get("phase") != "until_tool_was_held":
                raise ValueError("Unknown Strike frozen BC teacher phase")
            os.environ["EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT"] = str(teacher_path)
        else:
            os.environ.pop("EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT", None)
    print(f"[EVAL] {args.task} controller from checkpoint: {controller}", flush=True)
    return controller


def _selected_curriculum_stage() -> str:
    if args.curriculum_stage != "auto":
        return args.curriculum_stage
    if args.checkpoint != "auto" or not args.lineage_json or not args.individual_key:
        raise ValueError("Cannot infer curriculum stage without an automatic lineage checkpoint")
    with Path(args.lineage_json).open(encoding="utf-8") as file:
        individual = json.load(file)["lineage"][args.individual_key]
    stage = individual.get("metadata", {}).get("selected_curriculum_stage", "stage1")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unknown checkpoint curriculum stage: {stage}")
    return stage


def _metric(raw_env: Any, name: str, default: float = 0.0) -> float:
    value = raw_env.extras.get("log", {}).get(name, default)
    return _as_float(value)


def _list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().flatten().cpu()]


def _initial_geometry(task: str, raw_env: Any) -> dict[str, Any]:
    """Record the scene geometry needed to audit task reachability."""
    if task == "grasp":
        hand_pos = raw_env.hand.data.root_pos_w[0]
        hand_quat = raw_env.hand.data.root_quat_w[0]
        object_pos = raw_env.grasp_object.data.root_pos_w[0]
        relative_w = object_pos - hand_pos
        inverse_vec = -hand_quat[1:4]
        tangent = 2.0 * torch.cross(inverse_vec, relative_w, dim=-1)
        relative_local = relative_w + hand_quat[:1] * tangent + torch.cross(inverse_vec, tangent, dim=-1)
        center = torch.tensor(raw_env.cfg.visual_palm_region_center, device=raw_env.device)
        extents = torch.tensor(raw_env.cfg.visual_palm_region_half_extents, device=raw_env.device)
        return {
            "hand_root_world_m": _list(hand_pos),
            "object_world_m": _list(object_pos),
            "object_in_hand_local_m": _list(relative_local),
            "palm_center_local_m": _list(center),
            "palm_half_extents_m": _list(extents),
            "palm_axis_margin_m": _list(extents - torch.abs(relative_local - center)),
        }
    if task == "strike":
        cone = raw_env.cone_pos[0]
        tip = raw_env.cone_tip_pos[0]
        target = raw_env.strike_target_pos[0]
        return {
            "cone_root_world_m": _list(cone),
            "cone_tip_world_m": _list(tip),
            "strike_target_world_m": _list(target),
            "target_force_threshold_n": float(raw_env.cfg.success_force_threshold),
            "target_distance_threshold_m": float(raw_env.cfg.success_distance),
            "prestrike_hold_height_m": float(getattr(raw_env.cfg, "prestrike_hold_height", 0.0)),
        }
    return {}


def _prepare_morphology(output_dir: Path) -> MorphologyContext | None:
    """Build one morphology in an evaluation-local import override."""
    if not args.lineage_json:
        if args.individual_key:
            raise ValueError("--individual_key requires --lineage_json")
        return None
    if not args.individual_key:
        raise ValueError("--lineage_json requires --individual_key")

    code_root = Path(os.environ.get("EVOLUTION_CODE_ROOT", "/home/zjh/Evolution_PC"))
    sys.path.insert(0, str(code_root / "Isaaclab_other"))
    from code_to_urdf import generate_urdf_from_dict
    from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
    from mirror_agent import create_mirror_hand

    with Path(args.lineage_json).open(encoding="utf-8") as file:
        lineage = json.load(file)["lineage"]
    hand = lineage[args.individual_key]["urdf_info"]
    morphology_dir = output_dir / "morphology" / args.individual_key.replace("/", "_")
    right_urdf = morphology_dir / "right" / "urdf" / "current_agent.urdf"
    left_urdf = morphology_dir / "left" / "urdf" / "current_agent.urdf"
    right_urdf.parent.mkdir(parents=True, exist_ok=True)
    left_urdf.parent.mkdir(parents=True, exist_ok=True)
    generate_urdf_from_dict(hand, output_dir=str(morphology_dir / "right" / "meshes"), output_urdf=str(right_urdf))
    left_hand = create_mirror_hand(hand, f"{args.individual_key}_evaluation_left")
    generate_urdf_from_dict(left_hand, output_dir=str(morphology_dir / "left" / "meshes"), output_urdf=str(left_urdf))

    override_task_root = morphology_dir / "python_overrides" / "isaaclab_tasks" / "evolution_tasks"
    right_cfg = override_task_root / "current_right_hand" / "current_right_hand_cfg.py"
    left_cfg = override_task_root / "current_left_hand" / "current_left_hand_cfg.py"
    right_cfg.parent.mkdir(parents=True, exist_ok=True)
    left_cfg.parent.mkdir(parents=True, exist_ok=True)
    parse_urdf_and_generate_articulation_cfg(str(right_urdf), str(right_urdf), str(right_cfg))
    parse_urdf_and_generate_articulation_cfg(str(left_urdf), str(left_urdf), str(left_cfg))
    import isaaclab_tasks.evolution_tasks as evolution_tasks
    override_root = str(override_task_root)
    if override_root not in evolution_tasks.__path__:
        evolution_tasks.__path__ = [override_root, *list(evolution_tasks.__path__)]
    importlib.invalidate_caches()
    body_names = {
        link["name_code"]
        for link in hand.get("base_link", []) + hand.get("links", [])
    }
    contract_path = right_cfg
    spec = importlib.util.spec_from_file_location("evaluation_hand_cfg", contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated morphology config: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return MorphologyContext(override_task_root, body_names, module.MORPHOLOGY_CONTRACT)


def _evolved_fingertips(backup: MorphologyContext) -> list[str]:
    """Return the last actuated link body per finger, matching the training worker."""
    from policy_inheritance import distal_body_name

    fingertip_names = []
    available_joints = set(backup.morphology_contract["all_actuated_joints"])
    for finger_id in range(1, 6):
        body_name = distal_body_name(finger_id, available_joints)
        if body_name is None:
            raise ValueError(f"Morphology {args.individual_key} has no remaining body for finger {finger_id}.")
        fingertip_names.append(body_name)
    return fingertip_names


def _reload_task_modules(task: str) -> None:
    """Load generated hand modules before recreating the task configuration."""
    module_names = (
        "isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg",
        "isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg",
        TASKS[task][2],
    )
    importlib.invalidate_caches()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        importlib.reload(module)


def _configure_structure_adaptive_evaluation(task: str, env_cfg: Any, backup: MorphologyContext | None) -> None:
    """Bind the scene to the generated morphology exactly as the training worker does."""
    if backup is None:
        return
    fingertip_names = _evolved_fingertips(backup)
    available_joints = set(backup.morphology_contract["all_actuated_joints"])
    initial_state = env_cfg.robot_cfg.init_state
    requested_joint_pos = dict(initial_state.joint_pos or {})
    env_cfg.robot_cfg = env_cfg.robot_cfg.replace(
        init_state=initial_state.replace(
            joint_pos={name: value for name, value in requested_joint_pos.items() if name == ".*" or name in available_joints}
        )
    )
    right_cfg_path = backup.override_task_root / "current_right_hand" / "current_right_hand_cfg.py"
    spec = importlib.util.spec_from_file_location("evaluation_bound_hand_cfg", right_cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated morphology config: {right_cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env_cfg.robot_cfg = module.CURRENT_HAND_CFG.replace(prim_path=env_cfg.robot_cfg.prim_path).replace(
        init_state=env_cfg.robot_cfg.init_state
    )
    if hasattr(env_cfg, "fingertip_body_names"):
        env_cfg.fingertip_body_names = fingertip_names
    if task == "grasp":
        env_cfg.contact_sensor_cfg.filter_prim_paths_expr = [
            f"/World/envs/env_.*/LeftRobot/{name}" for name in fingertip_names
        ]
        env_cfg.thumb_contact_index = 0
        env_cfg.required_fingertip_count = len(fingertip_names)
    elif task == "branch":
        env_cfg.branch_contact_sensor_cfg.filter_prim_paths_expr = [
            f"/World/envs/env_.*/Robot/{name}" for name in fingertip_names
        ]
    elif task == "strike":
        env_cfg.tool_contact_sensor_cfg.filter_prim_paths_expr = [
            f"/World/envs/env_.*/RightRobot/{name}" for name in fingertip_names
        ]
    print(f"[EVAL] {task} adaptive fingertips: {fingertip_names}", flush=True)


def _task_evidence(task: str, raw_env: Any, reward: float) -> tuple[bool, dict[str, float | bool]]:
    """Use each environment's sparse success event, while retaining physical evidence."""
    if task == "grasp":
        contact_forces = _list(raw_env.full_hand_contact_forces[0])
        evidence = {
            "m1_any_fingertip_contact": bool(raw_env.any_fingertip_contact[0].item()),
            "m2_thumb_plus_other_contact": bool(raw_env.stage1_contact[0].item()),
            "m3_thumb_plus_long_finger_enclosure": bool(raw_env.full_hand_contact[0].item()),
            "milestone_hold_steps": _list(raw_env.milestone_streaks[0]),
            "milestones_claimed": [bool(item) for item in raw_env.milestone_claimed[0].tolist()],
            "m1_threshold_n": float(raw_env.cfg.m1_contact_force_threshold),
            "m2_threshold_n": float(raw_env.cfg.m2_contact_force_threshold),
            "m3_threshold_n": float(raw_env.cfg.m3_contact_force_threshold),
            "m3_long_finger_contacts": int(raw_env.m3_long_finger_contact_count[0].item()),
            "m3_required_long_finger_contacts": int(raw_env.cfg.m3_min_long_finger_contacts),
            "fingertip_contact_forces_n": contact_forces,
        }
    elif task == "branch":
        evidence = {
            "thumb_force_n": _metric(raw_env, "branch_thumb_force"),
            "other_finger_force_n": _metric(raw_env, "branch_other_finger_force"),
            "long_finger_contacts": _metric(raw_env, "branch_long_finger_contact_count"),
            "hold_steps": _as_float(raw_env.branch_success_streak[0]),
        }
    elif task == "forage":
        # Forage resets immediately after success.  The environment preserves
        # the pre-reset leaf distances explicitly for post-episode auditing.
        distances = raw_env.success_leaf_distances[0] if reward >= 999.0 else raw_env._leaf_distances()[0]
        evidence = {
            "leaf_one_distance_m": _as_float(distances[0]),
            "leaf_two_distance_m": _as_float(distances[1]),
            "both_leaves_cleared": reward >= 999.0 or bool(raw_env.success_achieved[0].item()),
        }
    else:
        evidence = {
            "strike_goal_distance_m": _metric(raw_env, "strike_goal_distance"),
            "strike_contact_force_n": _metric(raw_env, "strike_contact_force"),
            "tool_was_held": bool(raw_env.tool_was_held[0].item()),
            "thumb_tool_force_n": _as_float(raw_env.tool_fingertip_forces[0, 0]),
            "long_tool_contact_count": int((raw_env.tool_fingertip_forces[0, 1:] >= raw_env.cfg.tool_finger_contact_force_threshold).sum().item()),
            "grasp_hold_steps": int(raw_env.tool_grasp_streak[0].item()),
            "tool_attachment_error_m": _as_float(
                getattr(raw_env, "tool_attachment_error", torch.zeros(1, device=raw_env.device))[0]
            ),
        }
    # All four current tasks emit their sparse terminal reward only on a true
    # success event. This remains valid even when DirectRLEnv resets afterward.
    return reward >= 999.0, evidence


def main() -> None:
    if args.episodes != 1:
        raise ValueError("Run exactly one episode per IsaacLab process; use scripts/task_suite/run_reproducible_evaluation.sh for N episodes.")
    curriculum_stage = _selected_curriculum_stage()
    os.environ["EVOLUTION_CURRICULUM_STAGE"] = curriculum_stage
    os.environ["EVOLUTION_FORAGE_CURRICULUM_STAGE"] = curriculum_stage
    resume_path = retrieve_file_path(_resolve_checkpoint())
    controller = _configure_checkpoint_controller(resume_path)
    if os.environ.get("EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT"):
        if args.export_success_bc:
            raise ValueError("BC export cannot label PPO actions when the environment overrides the grasp phase")
        if args.replay_policy_trace:
            raise ValueError("Action replay cannot verify a trace when the environment overrides the grasp phase")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EVAL] Preparing morphology for {args.task}", flush=True)
    backup = _prepare_morphology(output_dir)
    print("[EVAL] Morphology configuration ready", flush=True)
    env = None
    try:
        import isaaclab_tasks  # noqa: F401

        env_id, module_name, _, _ = TASKS[args.task]
        importlib.import_module(module_name)
        if backup is not None:
            _reload_task_modules(args.task)
        env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1)
        _configure_structure_adaptive_evaluation(args.task, env_cfg, backup)
        env_cfg.seed = args.seed
        env_cfg.viewer.eye, env_cfg.viewer.lookat = CAMERA_VIEWS[args.task]
        env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
        agent_cfg = load_cfg_from_registry(env_id, "rl_games_cfg_entry_point")
        print(f"[EVAL] Creating {args.task} environment", flush=True)
        raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if args.record_video else None)
        if args.task == "strike" and controller == "strike_joint_target_v1":
            saved_contract = json.loads((Path(resume_path).parent.parent / "params" / "policy_contract.json").read_text(encoding="utf-8"))
            actual_joints = list(raw_env.unwrapped.hand.joint_names)
            if actual_joints != saved_contract.get("actual_joint_names"):
                raise ValueError("Strike checkpoint joint order differs from evaluation morphology")
        print(f"[EVAL] Environment created", flush=True)
        if args.record_video:
            set_camera_view(eye=env_cfg.viewer.eye, target=env_cfg.viewer.lookat, camera_prim_path="/OmniverseKit_Persp")
        if isinstance(raw_env.unwrapped, DirectMARLEnv):
            raw_env = multi_agent_to_single_agent(raw_env)
        rl_device = agent_cfg["params"]["config"]["device"]
        env = RlGamesVecEnvWrapper(
            raw_env,
            rl_device,
            agent_cfg["params"]["env"].get("clip_observations", math.inf),
            agent_cfg["params"]["env"].get("clip_actions", math.inf),
        )
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = 1
        runner = Runner(); runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player(); agent.restore(resume_path)
        videos_dir = output_dir / "successful_videos"
        if args.record_video:
            videos_dir.mkdir(exist_ok=True)
        episode_records: list[dict[str, Any]] = []
        started = time.monotonic()
        max_steps = args.max_steps or raw_env.unwrapped.max_episode_length
        replay_actions = None
        if args.replay_policy_trace:
            import numpy as np
            with np.load(args.replay_policy_trace) as dataset:
                replay_actions = dataset["submitted_actions"].copy()
            if replay_actions.ndim != 2 or replay_actions.shape[1] != raw_env.action_space.shape[-1]:
                raise ValueError("Replay action dimensions differ from the training environment")
            if args.replay_until_step:
                if args.replay_until_step > len(replay_actions):
                    raise ValueError("Replay prefix exceeds available actions")
            else:
                max_steps = min(max_steps, len(replay_actions))

        for _ in range(1):
            episode_index = args.episode_index
            episode_seed = args.seed
            torch.manual_seed(episode_seed)
            # Seed the underlying Gym environment before the wrapper obtains
            # its first observation for this episode.
            raw_env.reset(seed=episode_seed)
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            agent.reset()
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn: agent.init_rnn()
            temp_video = videos_dir / f".episode_{episode_index:03d}.mp4"
            writer = imageio.get_writer(temp_video, fps=args.video_fps, codec="libx264", quality=8) if args.record_video else None
            steps: list[dict[str, Any]] = []
            bc_observations = []
            bc_actions = []
            executed_actions = []
            strike_grip_action = None
            strike_teacher_wrist = torch.zeros((1, 3), device=raw_env.unwrapped.device)
            initial_geometry = _initial_geometry(args.task, raw_env.unwrapped)
            mesh_auditor = None
            if (args.replay_policy_trace or args.export_success_bc or args.audit_physics) and args.task == "grasp":
                from isaaclab_tasks.evolution_tasks.sphere_mesh_audit import SphereMeshAudit
                mesh_auditor = SphereMeshAudit(raw_env.unwrapped.hand)
            replay_joint_overshoot_max_rad = 0.0
            replay_joint_overshoot_steps = 0
            replay_mesh_penetration_steps = 0
            replay_mesh_min_clearance_m = float("inf")
            episode_success = False
            termination = "max_steps"
            try:
                for step in range(max_steps):
                    pre_state = None
                    if args.task == "strike":
                        physical = raw_env.unwrapped
                        pre_state = {
                            "tool_center_m": _list(physical.cone.data.root_pos_w[0]),
                            "tool_tip_m": _list(physical.cone_tip_pos[0]),
                            "hand_root_m": _list(physical.hand.data.root_pos_w[0]),
                            "target_m": _list(physical.strike_target_pos[0]),
                        }
                    if args.export_success_bc or args.export_rollout_debug:
                        bc_observations.append(raw_env.unwrapped._get_observations()["policy"][0].detach().cpu().numpy().astype("float32"))
                    with torch.inference_mode():
                        actions = (torch.as_tensor(replay_actions[step], device=raw_env.unwrapped.device).unsqueeze(0) if replay_actions is not None and (args.replay_until_step == 0 or step < args.replay_until_step) else agent.get_action(agent.obs_to_torch(obs), is_deterministic=True))
                        if actions.ndim == 1: actions = actions.unsqueeze(0)
                        if args.strike_wrist_teacher and args.task == "strike" and bool(raw_env.unwrapped.tool_was_held[0]):
                            physical = raw_env.unwrapped
                            if strike_grip_action is None:
                                strike_grip_action = actions[:, :20].clone()
                            if not args.strike_keep_policy_fingers:
                                actions[:, :20] = strike_grip_action
                            if args.strike_force_closure is not None:
                                actions[:, 15:20] = args.strike_force_closure
                            neutral = physical.hand.data.default_root_state[:, :3] + physical.scene.env_origins
                            root = physical.hand.data.root_pos_w
                            target = physical.strike_target_pos.clone()
                            target[:, 2] -= 0.012
                            tip_error = target - physical.cone_tip_pos
                            scale = physical.wrist_action_scale
                            desired = ((root + tip_error - neutral) / scale).clamp(-1.0, 1.0)
                            if args.strike_fixed_wrist_actions:
                                values = [float(value) for value in args.strike_fixed_wrist_actions.split(",")]
                                if len(values) != 3:
                                    raise ValueError("Strike fixed wrist requires three action values")
                                desired = torch.tensor(values, device=physical.device).view(1, 3).clamp(-1.0, 1.0)
                            strike_teacher_wrist = torch.maximum(
                                torch.minimum(desired, strike_teacher_wrist + args.strike_wrist_step),
                                strike_teacher_wrist - args.strike_wrist_step,
                            )
                            actions[:, 20:23] = strike_teacher_wrist
                        if args.export_success_bc or args.export_rollout_debug:
                            bc_actions.append(actions[0].detach().cpu().numpy().astype("float32"))
                        if pre_state is not None:
                            pre_state["submitted_wrist_action"] = _list(actions[0, -3:])
                            pre_state["submitted_finger_action_max_abs"] = float(actions[0, :20].abs().max())
                            pre_state["submitted_closure_actions"] = _list(actions[0, 15:20])
                        obs, reward, dones, _ = env.step(actions)
                        if args.export_success_bc or args.export_rollout_debug:
                            executed_actions.append(raw_env.unwrapped.raw_actions[0].detach().cpu().numpy().astype("float32"))
                    reward_value = _as_float(reward[0])
                    success_event, evidence = _task_evidence(args.task, raw_env.unwrapped, reward_value)
                    if args.replay_policy_trace or args.export_success_bc or args.audit_physics:
                        physical = raw_env.unwrapped
                        joints = physical.hand.data.joint_pos[0]
                        limits = physical.hand.root_physx_view.get_dof_limits()[0].to(joints.device)
                        overshoot = torch.maximum(limits[:, 0] - joints, joints - limits[:, 1]).clamp_min(0)
                        max_overshoot = float(overshoot.max())
                        replay_joint_overshoot_max_rad = max(replay_joint_overshoot_max_rad, max_overshoot)
                        replay_joint_overshoot_steps += int(max_overshoot > 0.02)
                        evidence["joint_overshoot_max_rad"] = max_overshoot
                        if max_overshoot > 0.02:
                            joint_id = int(overshoot.argmax())
                            evidence["joint_overshoot_name"] = physical.hand.joint_names[joint_id]
                            evidence["joint_position_rad"] = float(joints[joint_id])
                            evidence["joint_limits_rad"] = [float(v) for v in limits[joint_id]]
                            evidence["joint_target_rad"] = float(physical.cur_targets[0, joint_id]) if hasattr(physical, "cur_targets") else None
                        if mesh_auditor is not None:
                            clearance = mesh_auditor.measure(physical.grasp_object.data.root_pos_w[0].detach().cpu().numpy(), float(physical.cfg.grasp_object_cfg.spawn.radius))
                            value = float(clearance["clearance_m"])
                            replay_mesh_min_clearance_m = min(replay_mesh_min_clearance_m, value)
                            replay_mesh_penetration_steps += int(value < -1e-4)
                            evidence["source_mesh_clearance_m"] = value
                    episode_success |= success_event
                    if writer is not None:
                        frame = raw_env.render()
                        writer.append_data(frame[0] if isinstance(frame, (tuple, list)) else frame)
                    steps.append({
                        "pre_step_physics": pre_state,
                        "control_step": step,
                        "video_time_seconds": round(step / args.video_fps, 4),
                        "simulation_time_seconds": round((step + 1) * raw_env.unwrapped.step_dt, 4),
                        "reward": reward_value,
                        "success_event": success_event,
                        "evidence": evidence,
                    })
                    if success_event:
                        termination = "success"
                        break
                    if bool(dones[0].item()):
                        termination = "success" if episode_success else "terminated_or_timeout"
                        break
            finally:
                if writer is not None:
                    writer.close()
            record = {
                "episode": episode_index,
                "seed": episode_seed,
                "success": episode_success,
                "termination": termination,
                "steps": len(steps),
                "initial_geometry": initial_geometry,
                "strike_bc_teacher_checkpoint": os.environ.get("EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT") if args.task == "strike" else None,
                "joint_overshoot_max_rad": replay_joint_overshoot_max_rad if (args.replay_policy_trace or args.export_success_bc or args.audit_physics) else None,
                "joint_overshoot_steps": replay_joint_overshoot_steps if (args.replay_policy_trace or args.export_success_bc or args.audit_physics) else None,
                "source_mesh_penetration_steps": replay_mesh_penetration_steps if mesh_auditor is not None else None,
                "source_mesh_min_clearance_m": replay_mesh_min_clearance_m if mesh_auditor is not None else None,
                "trace": steps,
            }
            if episode_success and args.record_video:
                success_step = next(item["control_step"] for item in steps if item["success_event"])
                final_video = videos_dir / f"episode_{episode_index:03d}_seed_{episode_seed}_success_step_{success_step}.mp4"
                temp_video.replace(final_video)
                record["video"] = final_video.name
            elif args.keep_failure_videos and args.record_video:
                failure_dir = output_dir / "failure_videos"
                failure_dir.mkdir(exist_ok=True)
                final_video = failure_dir / f"episode_{episode_index:03d}_seed_{episode_seed}_failure.mp4"
                temp_video.replace(final_video)
                record["video"] = str(final_video.relative_to(output_dir))
            elif args.record_video:
                temp_video.unlink(missing_ok=True)
            if episode_success and args.export_success_bc:
                import numpy as np
                dataset_path = output_dir / "successful_policy_trace.npz"
                np.savez_compressed(
                    dataset_path,
                    observations_before_step=np.stack(bc_observations),
                    submitted_actions=np.stack(bc_actions),
                    executed_actions=np.stack(executed_actions),
                    actions_control_fingers=np.ones(len(bc_actions), dtype=bool),
                    scene_unmodified=np.ones(len(bc_actions), dtype=bool),
                )
                record["successful_policy_trace"] = str(dataset_path)
            if args.export_rollout_debug:
                import numpy as np
                debug_path = output_dir / "rollout_debug_trace.npz"
                np.savez_compressed(debug_path,
                    observations_before_step=np.stack(bc_observations),
                    submitted_actions=np.stack(bc_actions),
                    executed_actions=np.stack(executed_actions),
                    eligible_for_bc=np.array(False),
                )
                record["rollout_debug_trace"] = str(debug_path)
            episode_records.append(record)

        successes = sum(record["success"] for record in episode_records)
        report = {
            "task": args.task,
            "checkpoint": resume_path,
            "replay_policy_trace": args.replay_policy_trace,
            "strike_bc_teacher_checkpoint": os.environ.get("EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT") if args.task == "strike" else None,
            "replay_until_step": args.replay_until_step,
            "controller": controller,
            "strike_wrist_teacher": bool(args.strike_wrist_teacher),
            "curriculum_stage": curriculum_stage,
            "reset_dof_pos_noise": float(env_cfg.reset_dof_pos_noise),
            "morphology": {"lineage_json": args.lineage_json, "individual_key": args.individual_key},
            "seed_start": args.seed,
            "episodes": 1,
            "successes": successes,
            "success_rate": float(successes),
            "success_definition": "The environment's sparse terminal success event (reward >= 999).",
            "video_policy": "Videos disabled unless --record_video is supplied; failures also require --keep_failure_videos.",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "episode_records": episode_records,
        }
        with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    finally:
        if env is not None: env.close()
        app.close()


if __name__ == "__main__":
    main()
