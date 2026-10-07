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
    """Use the saved action contract before constructing a Branch environment."""
    if args.task != "branch":
        return None
    path = Path(checkpoint).parent.parent / "params" / "policy_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"Branch checkpoint has no policy contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("task") != TASKS["branch"][0]:
        raise ValueError(f"Branch checkpoint task mismatch: {path}")
    controller = contract.get("controller")
    modes = {"branch_joint_target_v1": "1", "cartesian_5finger_ik_v1": "0"}
    if controller not in modes:
        raise ValueError(f"Unknown Branch checkpoint controller: {controller}")
    os.environ["EVOLUTION_BRANCH_BC_MODE"] = modes[controller]
    print(f"[EVAL] Branch controller from checkpoint: {controller}", flush=True)
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
            initial_geometry = _initial_geometry(args.task, raw_env.unwrapped)
            episode_success = False
            termination = "max_steps"
            try:
                for step in range(max_steps):
                    with torch.inference_mode():
                        actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                        if actions.ndim == 1: actions = actions.unsqueeze(0)
                        obs, reward, dones, _ = env.step(actions)
                    reward_value = _as_float(reward[0])
                    success_event, evidence = _task_evidence(args.task, raw_env.unwrapped, reward_value)
                    episode_success |= success_event
                    if writer is not None:
                        frame = raw_env.render()
                        writer.append_data(frame[0] if isinstance(frame, (tuple, list)) else frame)
                    steps.append({
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
            episode_records.append(record)

        successes = sum(record["success"] for record in episode_records)
        report = {
            "task": args.task,
            "checkpoint": resume_path,
            "controller": controller,
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
