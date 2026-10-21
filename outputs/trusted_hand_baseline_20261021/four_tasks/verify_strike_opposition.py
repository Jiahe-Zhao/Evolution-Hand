"""Reproducible, per-episode evaluation for Evolution RL-Games policies."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

sys.path.insert(0,'/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
from isaaclab.app import AppLauncher

# This entry point lives beside the mechanical asset tools, while the shared
# evaluation helpers live in the repository's task suite.
sys.path.insert(0, '/home/zjh/Evolution_PC/evolution_tasks/task_suite')
from task_registry import TASKS


parser = argparse.ArgumentParser(description="Evaluate one fixed morphology and policy over fixed seeds.")
parser.add_argument("--mechanical_joint_actions", action="store_true")
parser.add_argument("--bc_manifest")
parser.add_argument("--bc_only", action="store_true")
parser.add_argument("--train_iterations", type=int, default=0)
parser.add_argument("--train_num_envs", type=int, default=4096)
parser.add_argument("--train_smoke", action="store_true")
parser.add_argument("--train_from_scratch", action="store_true", help="Initialize PPO weights when morphology joints differ from the reference checkpoint")
parser.add_argument("--train_auto_inherit", action="store_true", help="Inherit common observation/action channels by semantic joint names")
parser.add_argument("--physics_batch_probe_envs", type=int, default=0, help="Run a separate batched physics/reset probe, not a policy-success evaluation")
parser.add_argument("--physics_batch_probe_zero_actions", action="store_true", help="Diagnostic: hold zero actions for the full batch probe")
parser.add_argument("--palm_coordination", action="store_true", help="Coordinate four palm supports; retain independent distal commands")
parser.add_argument("--mechanical_wrist_speed", type=float, help="Diagnostic physical wrist speed cap in m/s; changes controller contract")
parser.add_argument("--compliant_palm", action="store_true", help="Include the physical FEM palm in the task scene")
parser.add_argument("--legacy_teacher_urdf", help="Exact morphology from the frozen teacher verification record")
parser.add_argument("--legacy_source_urdf", required=True, help="Archived source morphology used by the old policy")
parser.add_argument("--mechanical_asset_dir", required=True, help="Directory containing right/urdf and left/urdf/current_agent.urdf")
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--checkpoint", default="auto")
parser.add_argument("--curriculum_stage", choices=("auto", "stage1", "stage2"), default="auto")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--episodes", type=int, default=1, help="One process evaluates one episode; use run_reproducible_evaluation.sh for N episodes.")
parser.add_argument("--seed", type=int, default=7, help="First deterministic episode seed.")
parser.add_argument("--replay_until_step", type=int, default=0, help="Diagnostic: use saved actions through this step, then run the checkpoint policy.")
parser.add_argument("--replay_policy_trace", help="Replay a saved successful policy action trace with identical seed and scene.")
parser.add_argument("--audit_physics", action="store_true", help="Audit physical joint limits and Grasp source-mesh clearance.")
parser.add_argument("--forage_solver_iterations", type=int, help="Diagnostic Forage articulation solver accuracy; changes physics contract.")
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
_worker_mode = os.environ.get("EVOLUTION_ISAAC_WORKER_MODE") == "1"
if os.environ.get("GRAPH_SKIP_LOG_SCAN") == "1":
    # Diagnostic-process-only: historical training logs contain no task modules.
    import omni.ext._impl.fast_importer as _fast_importer
    _original_fast_walk = _fast_importer._fast_walk
    _excluded_logs = '/home/zjh/Evolution_PC/evolution_tasks/logs'
    def _diagnostic_fast_walk(path, *walk_args, **walk_kwargs):
        if os.path.realpath(path) == _excluded_logs:
            return
        yield from _original_fast_walk(path, *walk_args, **walk_kwargs)
    _fast_importer._fast_walk = _diagnostic_fast_walk
app = None if _worker_mode else AppLauncher(args).app

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
    path = Path(checkpoint).parent.parent / "params" / "policy_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"{args.task} checkpoint has no policy contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("task") != TASKS[args.task][0]:
        raise ValueError(f"{args.task} checkpoint task mismatch: {path}")
    controller = contract.get("controller")
    if controller == 'joint_graph_v2':
        if contract.get('version') != 2 or contract.get('mechanical_scene',{}).get('version') != 5:
            raise ValueError('Graph checkpoint lacks v2/v5 contract')
        return controller
    modes = ({"branch_joint_target_v1": "1", "cartesian_5finger_ik_v1": "0"}
             if args.task == "branch" else
             {"strike_joint_target_v1": "1", "cartesian_5finger_ik_v1": "0"} if args.task == "strike" else
             {"cartesian_5finger_ik_v1": "0", f"{args.task}_joint_target_v1":"1"})
    if controller not in modes:
        raise ValueError(f"Unknown {args.task} checkpoint controller: {controller}")
    if args.task in {"branch", "strike"}:
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
    if controller in ('grasp_joint_target_v1','forage_joint_target_v1'):
        args.mechanical_joint_actions=True
    if args.task in ('grasp','forage') and args.mechanical_joint_actions:
        target_controller=f'{args.task}_joint_target_v1'
        if controller != target_controller and not (args.train_from_scratch or args.physics_batch_probe_envs):
            raise ValueError('New joint action interface requires BC/scratch initialization, not an old Cartesian policy')
        controller=target_controller
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

    import xml.etree.ElementTree as ET
    source_thumb = ET.parse(args.legacy_source_urdf).getroot().find("joint[@name='link_1_thumb_spread_joint']/limit")
    global mechanical_source_lower, mechanical_target_upper
    mechanical_source_lower = float(source_thumb.get('lower'))
    if float(source_thumb.get('upper')) != 0 or mechanical_source_lower >= 0:
        raise ValueError('Expected legacy thumb range [negative, 0]')
    right_urdf = Path(args.mechanical_asset_dir).resolve() / 'right/urdf/current_agent.urdf'
    left_urdf = Path(args.mechanical_asset_dir).resolve() / 'right/urdf/current_agent.urdf'
    target_thumb = ET.parse(right_urdf).getroot().find("joint[@name='link_1_thumb_spread_joint']/limit")
    mechanical_target_upper = float(target_thumb.get('upper'))
    if float(target_thumb.get('lower')) != 0 or mechanical_target_upper <= 0:
        raise ValueError('Expected mechanical thumb range [0, positive]')

    override_task_root = morphology_dir / "python_overrides" / "isaaclab_tasks" / "evolution_tasks"
    from isaaclab_tasks.evolution_tasks.mechanical_hand_interface import discover
    source_contract = discover(morphology_dir / "right" / "urdf" / "current_agent.urdf")
    mechanical_right = Path(args.mechanical_asset_dir).resolve() / "right/urdf/current_agent.urdf"
    mechanical_left = Path(args.mechanical_asset_dir).resolve() / "right/urdf/current_agent.urdf"
    for mechanical_urdf in (mechanical_right, mechanical_left):
        if not mechanical_urdf.is_file():
            raise FileNotFoundError(f"Missing mechanical phenotype: {mechanical_urdf}")
        target_contract = discover(mechanical_urdf)
        if (target_contract["canonical_joint_names"] != source_contract["canonical_joint_names"]
                or target_contract["tip_links"] != source_contract["tip_links"]):
            raise ValueError(f"Mechanical phenotype does not match bone genotype: {mechanical_urdf}")

    right_cfg = override_task_root / "current_right_hand" / "current_right_hand_cfg.py"
    left_cfg = override_task_root / "current_left_hand" / "current_left_hand_cfg.py"
    right_cfg.parent.mkdir(parents=True, exist_ok=True)
    left_cfg.parent.mkdir(parents=True, exist_ok=True)
    parse_urdf_and_generate_articulation_cfg(str(mechanical_right), str(mechanical_right), str(right_cfg))
    parse_urdf_and_generate_articulation_cfg(str(mechanical_left), str(mechanical_left), str(left_cfg))
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
    thumb_name = 'link_1_thumb_spread_joint'
    if thumb_name in requested_joint_pos:
        requested_joint_pos[thumb_name] *= mechanical_target_upper / mechanical_source_lower
    env_cfg.robot_cfg = env_cfg.robot_cfg.replace(
        init_state=initial_state.replace(
            joint_pos={name: value for name, value in requested_joint_pos.items() if name == ".*" or name in available_joints}
        )
    )
    side = "right" if args.task == "grasp" and args.mechanical_joint_actions else ("left" if "LeftRobot" in env_cfg.robot_cfg.prim_path else "right")
    bound_cfg_path = backup.override_task_root / f"current_{side}_hand" / f"current_{side}_hand_cfg.py"
    spec = importlib.util.spec_from_file_location("evaluation_bound_hand_cfg", bound_cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generated morphology config: {bound_cfg_path}")
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
        env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=args.physics_batch_probe_envs or (args.train_num_envs if args.train_iterations else 1))
        _configure_structure_adaptive_evaluation(args.task, env_cfg, backup)
        if args.forage_solver_iterations is not None:
            if args.task != "forage" or not 8 <= args.forage_solver_iterations <= 255:
                raise ValueError("Forage solver diagnostic requires 8..255 iterations")
            env_cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.forage_solver_iterations
            env_cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = min(args.forage_solver_iterations, 32)
        from mechanical_scene_contract import apply_mechanical_scene
        apply_mechanical_scene(env_cfg,args.task,output_dir)
        env_cfg.seed = args.seed
        env_cfg.viewer.eye, env_cfg.viewer.lookat = CAMERA_VIEWS[args.task]
        env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
        import yaml
        saved_agent_config=Path(resume_path).parent.parent/"params/agent.yaml"
        agent_cfg = yaml.safe_load(saved_agent_config.read_text())
        print(f"[EVAL] Creating {args.task} environment", flush=True)
        if args.task == 'strike':
            reset_contract = output_dir.resolve()/'mechanical_reset_transfer.json'
            reset_contract.write_text(json.dumps({'source_lower':mechanical_source_lower,
                'target_upper':mechanical_target_upper,
                'source_urdf_sha256':hashlib.sha256(Path(args.legacy_source_urdf).read_bytes()).hexdigest(),
                'target_urdf_sha256':hashlib.sha256(Path(env_cfg.robot_cfg.spawn.asset_path).read_bytes()).hexdigest(),
                'scope':'Physical reset and tool-placement probe; independent of BC teacher'},indent=2))
            os.environ['EVOLUTION_MECHANICAL_RESET_TRANSFER'] = str(reset_contract)
        else:
            os.environ.pop('EVOLUTION_MECHANICAL_RESET_TRANSFER',None)
        if os.environ.get('EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT'):
            from isaaclab_tasks.evolution_tasks.task_strike.strike_bc_teacher import morphology_sha256
            from isaaclab_tasks.evolution_tasks.mechanical_policy_transfer import ThumbTransfer
            if not args.legacy_teacher_urdf:
                raise ValueError('Strike teacher transfer requires its separately verified source URDF')
            teacher_path = Path(os.environ['EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT'])
            teacher_contract = json.loads((teacher_path.parent.parent / 'params/policy_contract.json').read_text())
            names = list(env_cfg.actuated_joint_names)
            mapping = ThumbTransfer(mechanical_source_lower, mechanical_target_upper,
                                    (names.index('link_1_thumb_spread_joint') if 'link_1_thumb_spread_joint' in names else None), len(names), 23,
                                    teacher_contract['actual_joint_names'].index('link_1_thumb_spread_joint'))
            transfer_file = output_dir.resolve() / 'strike_teacher_transfer.json'
            transfer_file.write_text(json.dumps({
                'checkpoint_sha256': hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                'source_urdf': str(Path(args.legacy_teacher_urdf).resolve()),
                'target_morphology_sha256': morphology_sha256(env_cfg.robot_cfg.spawn.asset_path),
                'mapping': mapping.__dict__,
                'status': 'experimental_transfer_not_verified_task_success'
            }, indent=2))
            os.environ['EVOLUTION_STRIKE_MECHANICAL_TRANSFER'] = str(transfer_file)
        else:
            os.environ.pop('EVOLUTION_STRIKE_MECHANICAL_TRANSFER', None)
        if args.compliant_palm:
            env_cfg.scene.replicate_physics = False
            from compliant_palm_task import install_palm_hooks
            install_palm_hooks(env_id, args.mechanical_asset_dir, output_dir, args.mechanical_wrist_speed, args.palm_coordination)
        mechanical_urdf = Path(env_cfg.robot_cfg.spawn.asset_path).resolve()
        mechanical_root = ET.parse(mechanical_urdf).getroot()
        rigid_palm = mechanical_root.find("link[@name='link_palm_chassis']")
        if rigid_palm is None or not rigid_palm.findall("collision"):
            raise ValueError("Mechanical hand URDF must include a collidable link_palm_chassis")
        palm_contract = {
            "mode": "rigid_morphology_adaptive_palm_v1",
            "urdf": str(mechanical_urdf),
            "collision_count": len(rigid_palm.findall("collision")),
            "fixed_to_wrist_and_parent_of_digit_roots": True,
            "mechanical_wrist_speed_m_s": args.mechanical_wrist_speed,
            "sha256": hashlib.sha256(mechanical_urdf.read_bytes()).hexdigest(),
        }
        (output_dir / "palm_assembly_contract.json").write_text(json.dumps(palm_contract, indent=2))
        from joint_graph_env import install as install_graph
        if os.environ.get('GRAPH_PROBE_HALF_DT') == '1':
            env_cfg.sim.dt *= 0.5
            env_cfg.decimation *= 2
        if os.environ.get('GRAPH_TEACHER')=='1':
            env_cfg.episode_length_s=float(os.environ.get("GRAPH_TEACHER_EPISODE_S","40.0"))
        driven_contract=None
        if os.environ.get('GRAPH_DRIVEN_WRIST')=='1':
            if args.task not in ('strike','forage'):raise ValueError('Wrist only for moving-wrist tasks')
            from driven_wrist_adapter import configure
            driven_contract=configure(env_cfg,mechanical_urdf,output_dir)
            (output_dir/'driven_wrist_contract.json').write_text(json.dumps(driven_contract,indent=2))
        graph = install_graph(env_id, env_cfg, args.task, mechanical_urdf, driven_contract=driven_contract, wrist_speed_m_s=float(os.environ.get("GRAPH_WRIST_SPEED","0.006")))
        if os.environ.get('GRAPH_STRIKE_SUPPORT')=='1':
            from strike_support_v5 import install as install_support
            install_support(env_id,env_cfg)
        from baseline_hooks import install as install_baseline
        install_baseline(env_id,env_cfg,output_dir,args.task)
        raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if args.record_video else None)
        raw = raw_env.unwrapped
        if os.environ.get('BASELINE_KINEMATIC_SYNC')=='1':
            old_reset=raw._reset_idx
            def sync_reset(ids):
                old_reset(ids)
                from isaaclab.utils.math import convert_quat
                for obj in raw.scene.rigid_objects.values():
                    props=obj.cfg.spawn.rigid_props
                    if props and props.kinematic_enabled:
                        poses=obj.data.root_state_w[:,:7].clone();poses[:,3:]=convert_quat(poses[:,3:],to='xyzw')
                        obj.root_physx_view.set_kinematic_targets(poses,obj._ALL_INDICES)
            raw._reset_idx=sync_reset
        obs, _ = raw_env.reset(seed=args.seed)
        if args.task=='strike':
            from strike_opposition import place
            place(raw,output_dir)
        if os.environ.get('BASELINE_RESET_SETTLE')=='1':
            # Synchronize kinematic pose history during reset, then restore every initial state.
            saved_root=raw.hand.data.root_state_w.clone();saved_q=raw.hand.data.joint_pos.clone();saved_dq=raw.hand.data.joint_vel.clone();saved_targets=raw.prev_targets.clone()
            saved_objects={name:obj.data.root_state_w.clone() for name,obj in raw.scene.rigid_objects.items()}
            raw.scene.write_data_to_sim();raw.sim.step(render=False);raw.scene.update(raw.cfg.sim.dt)
            raw.hand.write_root_state_to_sim(saved_root);raw.hand.write_joint_state_to_sim(saved_q,saved_dq);raw.hand.set_joint_position_target(saved_targets)
            for name,state in saved_objects.items():raw.scene.rigid_objects[name].write_root_state_to_sim(state)
            raw.scene.write_data_to_sim();raw.sim.forward();raw.scene.update(0.)
            (output_dir/'reset_settle.json').write_text(json.dumps({'physics_steps_before_episode':1,'restored_all_hand_and_object_states':True,'no_object_pose_writes_during_episode':True}))
        if driven_contract and args.task=='strike':
            from audit_runtime_physics import run as audit_runtime
            audit_runtime(raw,output_dir)
        scene_path = output_dir/'mechanical_scene_contract.json'
        graph_scene = json.loads(scene_path.read_text())
        previous_implementation = graph_scene['implementation_sha256']
        graph_scene['version'] = 5
        graph_scene['effective_settings']['joint_graph_controller'] = raw.graph_contract
        graph_scene['effective_settings']['physics_step'] = {'dt':env_cfg.sim.dt,'decimation':env_cfg.decimation}
        graph_scene['effective_settings']['wrist_contact_gate'] = False
        graph_scene['effective_settings']['long_finger_spread_locked'] = False
        graph_scene['effective_settings']['observations_include_contact_and_commands'] = True
        graph_scene['implementation_sha256'] = hashlib.sha256((previous_implementation +
            hashlib.sha256(Path(__file__).with_name('joint_graph_env.py').read_bytes()).hexdigest() +
            hashlib.sha256(Path(__file__).with_name('joint_graph_control.py').read_bytes()).hexdigest()).encode()).hexdigest()
        if args.task=='branch':
            graph_scene['effective_settings']['branch_reset_clearance']={'geometry':'imported_collision_hulls_including_fingertips','margin_m':0.001,'max_offset_m':0.016}
            graph_scene['implementation_sha256']=hashlib.sha256((graph_scene['implementation_sha256']+
                hashlib.sha256(Path(__file__).with_name('branch_reset_clearance_v5.py').read_bytes()).hexdigest()+
                hashlib.sha256(Path(__file__).with_name('imported_hand_geometry.py').read_bytes()).hexdigest()).encode()).hexdigest()
            (output_dir/'branch_reset_clearance.json').write_text(json.dumps(raw.branch_clearance_report,indent=2))
        scene_path.write_text(json.dumps(graph_scene,indent=2))
        if os.environ.get('GRAPH_STRIKE_SUPPORT')=='1':
            from strike_support_v5 import audit
            (output_dir/'support_collision_audit.json').write_text(json.dumps(audit(raw),indent=2))
            graph_scene['effective_settings']['tool_support']=raw.tool_support_contract
            graph_scene['implementation_sha256']=hashlib.sha256((graph_scene['implementation_sha256']+hashlib.sha256(Path(__file__).with_name('strike_support_v5.py').read_bytes()).hexdigest()).encode()).hexdigest()
            scene_path.write_text(json.dumps(graph_scene,indent=2))
        from strike_opposition_rollout import run
        run(raw_env,graph,output_dir,args,_task_evidence)
        os._exit(0)
        if os.environ.get('GRAPH_LOAD_PLAN')=='1':
            from plan_load_pose import run as plan_load
            plan_load(raw,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_POLICY_EVAL')=='1':
            from evaluate_graph_policy import run
            run(raw_env,resume_path,output_dir,args.task,args.seed)
            os._exit(0)
        if os.environ.get('GRAPH_PAD_CALIBRATE')=='1':
            from calibrate_pad_closure import run
            run(raw,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_LOAD_EXECUTE')=='1':
            from execute_load_path import run as execute_load
            from calibrate_pad_closure import run as calibrate_load
            from strike_feedback_teacher_v5 import rollout
            os.environ['GRAPH_PAD_PLAN']=os.environ['GRAPH_LOAD_PATH']
            calibrate_load(raw,output_dir)
            os.environ['GRAPH_PAD_CALIBRATION']=str(output_dir/'pad_closure.json')
            modes=['withdraw','lift'] if os.environ.get('GRAPH_LOAD_TEST_BOTH')=='1' else [os.environ.get('GRAPH_LOAD_MODE','lift')]
            reports={}
            for mode in modes:
                destination=output_dir/mode;destination.mkdir(exist_ok=True)
                os.environ['GRAPH_LOAD_MODE']=mode
                raw_env.reset(seed=args.seed)
                path_result=execute_load(raw_env,destination)
                reports[mode]={'path':path_result}
                if path_result['completed']:reports[mode]['teacher']=rollout(raw_env,graph,destination)
                (output_dir/'teacher_report.json').write_text(json.dumps(reports,indent=2))
            os._exit(0)
        if os.environ.get('GRAPH_PATH_EXECUTE')=='1':
            from execute_thumb_path import run
            path_result=run(raw_env,output_dir)
            if os.environ.get('GRAPH_PATH_GRASP')=='1' and path_result['completed']:
                from strike_feedback_teacher_v5 import rollout
                rollout(raw_env,graph,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_THUMB_PATH')=='1':
            from audit_thumb_paths import run
            run(raw,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_THUMB_SEARCH')=='1':
            from search_thumb_pregrasp import run
            run(raw,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_JACOBIAN_AUDIT')=='1':
            from probe_tip_jacobian import run
            run(raw,output_dir)
            os._exit(0)
        if os.environ.get('GRAPH_TEACHER')=='1':
            graph_scene['effective_settings']['diagnostic_episode_length_s']=40.0
            scene_path.write_text(json.dumps(graph_scene,indent=2))
            from strike_feedback_teacher_v5 import rollout
            rollout(raw_env,graph,output_dir)
            os._exit(0)
        tool_start=raw.cone.data.root_state_w.clone() if args.task=='strike' else None
        observations = [tuple(obs['policy'].shape)]
        control = raw.graph_control
        q = raw.hand.data.joint_pos.index_select(-1, control.indices)
        neutral = 2*(q-control.lower)/(control.upper-control.lower)-1
        initial = raw.hand.data.joint_pos.clone()
        diagnostic = os.environ.get('GRAPH_BRANCH_DIAG','motion')
        if args.task=='branch' and diagnostic.startswith('no_object'):
            state = raw.branch.data.root_state_w.clone()
            state[:,2] += 3.0
            raw.branch.write_root_state_to_sim(state)
            raw.sim.forward();raw.scene.update(dt=0.0)
        max_motion = torch.zeros_like(initial)
        peak_events = []
        resets = 0
        max_overshoot = 0.0
        overshoot_by_joint = torch.zeros_like(initial)
        for step in range(120):
            actions = torch.zeros((raw.num_envs,env_cfg.action_space),device=raw.device)
            phase = torch.arange(len(graph.joints),device=raw.device)*0.37+step*0.06
            actions[:,:len(graph.joints)] = (neutral+0.04*torch.sin(phase)).clamp(-1,1)
            if diagnostic.endswith('static'):
                actions[:,:len(graph.joints)] = neutral
            obs,reward,terminated,truncated,info = raw_env.step(actions)
            resets += int((terminated|truncated).sum())
            if not bool(torch.isfinite(obs['policy']).all()): raise RuntimeError('Non-finite graph observations')
            max_motion = torch.maximum(max_motion,(raw.hand.data.joint_pos-initial).abs())
            limits = raw.hand.root_physx_view.get_dof_limits().to(raw.device)
            violation = torch.maximum(limits[...,0]-raw.hand.data.joint_pos,raw.hand.data.joint_pos-limits[...,1]).clamp_min(0)
            overshoot_by_joint = torch.maximum(overshoot_by_joint,violation)
            if float(violation.max()) > max_overshoot:
                flat = int(violation.argmax());ei,ji=divmod(flat,violation.shape[1])
                peak_events.append(dict(step=step,env=ei,joint=raw.hand.joint_names[ji],
                    position=float(raw.hand.data.joint_pos[ei,ji]),target=float(raw.prev_targets[ei,ji]),
                    velocity=float(raw.hand.data.joint_vel[ei,ji]),lower=float(limits[ei,ji,0]),upper=float(limits[ei,ji,1]),
                    overshoot=float(violation.max())))
            max_overshoot = max(max_overshoot,float(violation.max()))
        report = dict(diagnostic=diagnostic,reset_count=resets,peak_events=peak_events,physics_dt=env_cfg.sim.dt, decimation=env_cfg.decimation, task=args.task, controller='joint_graph_v2', joint_count=len(graph.joints),
                      action_dimension=env_cfg.action_space, observation_dimension=env_cfg.observation_space,
                      steps=120, finite_observations=True, max_overshoot_rad=max_overshoot,
                      moved_joints=int((max_motion.max(0).values>1e-4).sum()),
                      scope='task environment control/observation smoke; not learned task success',
                      overshoot_by_joint={name:float(overshoot_by_joint[:,i].max()) for i,name in enumerate(raw.hand.joint_names)},
                      effective_joint_velocity_limits=raw.graph_contract['effective_joint_velocity_limits'])
        if tool_start is not None:
            report['tool_displacement_m']=(raw.cone.data.root_pos_w-tool_start[:,:3]).cpu().tolist()
            report['tool_final_velocity']=raw.cone.data.root_state_w[:,7:].cpu().tolist()
        (output_dir/'joint_graph_probe.json').write_text(json.dumps(report,indent=2))
        (output_dir/'joint_graph_contract.json').write_text(json.dumps(raw.graph_contract,indent=2))
        print('JOINT_GRAPH_PROBE',json.dumps(report),flush=True)
        if os.environ.get('GRAPH_PROBE_PPO') == '1':
            from mechanical_ppo_v5 import train_mechanical
            class PhysicalIdentity:
                def observation_for_source_policy(self, obs): return obs
                def action_for_target_hand(self, actions): return actions
            train_mechanical(raw_env,PhysicalIdentity(),resume_path,output_dir,2,
                             smoke=True,inherit_weights=os.environ.get('GRAPH_PROBE_INHERIT')=='1',task=args.task)
        # Isolated diagnostic: avoid known Isaac shutdown hangs after report.
        os._exit(0)
        if os.environ.get("EVOLUTION_DIAG_SCENE_START_ONLY") == "1":
            print("[MECHANICAL_SCENE_START] passed", flush=True)
            raw_env.close()
            return
        if args.physics_batch_probe_envs:
            if args.physics_batch_probe_envs < 2:
                raise ValueError('Batch probe requires at least two environments')
            physical = raw_env.unwrapped
            raw_env.reset(seed=args.seed)
            assert physical.hand.data.joint_pos.shape[0] == args.physics_batch_probe_envs
            actions = torch.zeros(raw_env.action_space.shape, device=physical.device)
            reset_leak = 0.0
            for step_index in range(300):
                if step_index >= 100 and not args.physics_batch_probe_zero_actions:
                    phase = (step_index - 100) / 24.0
                    actions[:, :20] = 0.35 * torch.sin(
                        torch.arange(20, device=physical.device).float() * 0.31 + phase
                    )
                    if actions.shape[1] > 20:
                        actions[:, 20:] = 0.15 * torch.sin(
                            torch.arange(actions.shape[1] - 20, device=physical.device).float() + phase
                        )
                _, reward, _, _, _ = raw_env.step(actions)
                if not bool(torch.isfinite(reward).all()):
                    raise RuntimeError('Nonfinite batch reward')
                if step_index == 100:
                    before = {
                        "joint_pos": physical.hand.data.joint_pos.clone(),
                        "root_pos": physical.hand.data.root_pos_w.clone(),
                    }
                    physical._reset_idx(torch.tensor([1], device=physical.device))
                    physical.sim.forward()
                    physical.hand.update(0.)
                    after = {
                        "joint_pos": physical.hand.data.joint_pos,
                        "root_pos": physical.hand.data.root_pos_w,
                    }
                    untouched = torch.arange(physical.num_envs,device=physical.device) != 1
                    reset_leak = max(float((after[key][untouched]-before[key][untouched]).abs().max())
                                      for key in before)
                    if reset_leak > 1e-6:
                        raise RuntimeError(f'Partial reset changed unrelated mechanical hands: {reset_leak}')
                for name in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w"):
                    value = getattr(physical.hand.data, name)
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError(f'Nonfinite mechanical hand state: {name}')
            passed = True
            report = {'probe_only':True,'task':args.task,'num_envs':physical.num_envs,
                      'steps':300,'motion_probe':('zero_actions_v1' if args.physics_batch_probe_zero_actions else 'bounded_sinusoidal_v1'),
                      'partial_reset_other_env_delta_m':reset_leak,
                      'rigid_palm':palm_contract,'all_hand_states_finite':True,'passed':passed}
            (output_dir/'physics_batch_probe.json').write_text(json.dumps(report,indent=2))
            print('[MECHANICAL_BATCH_PROBE]',json.dumps(report),flush=True)
            raw_env.close()
            return
        if (args.task == "strike" and controller == "strike_joint_target_v1"
                and not (args.train_iterations and (args.train_from_scratch or args.train_auto_inherit))):
            saved_contract = json.loads((Path(resume_path).parent.parent / "params" / "policy_contract.json").read_text(encoding="utf-8"))
            actual_joints = list(raw_env.unwrapped.hand.joint_names)
            if actual_joints != saved_contract.get("actual_joint_names"):
                raise ValueError("Strike checkpoint joint order differs from evaluation morphology")
        from isaaclab_tasks.evolution_tasks.mechanical_policy_transfer import ThumbTransfer
        physical = raw_env.unwrapped
        thumb = 'link_1_thumb_spread_joint'
        canonical = list(physical.canonical_joint_names)
        action_dim = raw_env.action_space.shape[-1]
        if controller == 'cartesian_5finger_ik_v1':
            action_index = None
        elif controller == 'strike_joint_target_v1':
            action_index = list(physical.hand.joint_names).index(thumb)
        elif controller in ('branch_joint_target_v1','grasp_joint_target_v1','forage_joint_target_v1'):
            thumb_ids = physical.cartesian_ik.joint_ids[0].tolist()
            action_index = thumb_ids.index(list(physical.hand.joint_names).index(thumb))
        else:
            raise ValueError('Unbound mechanical controller')
        transfer = ThumbTransfer(mechanical_source_lower, mechanical_target_upper,
                                 (canonical.index(thumb) if thumb in canonical else None), len(canonical), action_dim, action_index)
        saved_policy_contract = json.loads((Path(resume_path).parent.parent/'params/policy_contract.json').read_text())
        saved_scene=saved_policy_contract.get('mechanical_scene')
        if saved_scene and not args.train_auto_inherit and not args.train_from_scratch:
            actual_scene=json.loads((output_dir/'mechanical_scene_contract.json').read_text())
            for field in ('version','task','effective_settings','curriculum_stage','implementation_sha256'):
                if saved_scene.get(field) != actual_scene.get(field):
                    raise ValueError('Saved training scene differs from evaluation scene: '+field)
        trained_adapter = saved_policy_contract.get('mechanical_adapter')
        adapter_compatible = True
        if trained_adapter:
            expected_adapter = {
                'observation_convention':'legacy_semantic_after_physical_clip',
                'mapping':transfer.__dict__,
                'rigid_palm_mode':palm_contract['mode'],
                'wrist_speed_m_s':args.mechanical_wrist_speed,
                'target_urdf_sha256':hashlib.sha256(Path(physical.hand.cfg.spawn.asset_path).read_bytes()).hexdigest()}
            adapter_compatible = ({k: v for k, v in trained_adapter.items() if k != 'target_urdf_sha256'}
                                  == {k: v for k, v in expected_adapter.items() if k != 'target_urdf_sha256'})
            if not args.train_auto_inherit and not args.train_from_scratch and trained_adapter != expected_adapter:
                raise ValueError('Mechanical checkpoint and evaluation physics/control contract differ')
        (output_dir / 'mechanical_transfer.json').write_text(json.dumps({
            'legacy_urdf_sha256': hashlib.sha256(Path(args.legacy_source_urdf).read_bytes()).hexdigest(),
            'checkpoint_sha256': hashlib.sha256(Path(resume_path).read_bytes()).hexdigest(),
            'target_urdf_sha256': hashlib.sha256(Path(physical.hand.cfg.spawn.asset_path).read_bytes()).hexdigest(),
            'controller': controller, 'mapping': transfer.__dict__,
            'semantics': 'normalized excursion transfer; does not preserve Cartesian thumb trajectories'
        }, indent=2))
        if args.train_iterations:
            if args.physics_batch_probe_envs:
                raise ValueError('Batch physics probe cannot train PPO')
            reference_joints = saved_policy_contract.get('actual_joint_names')
            target_joints = list(physical.hand.joint_names)
            if not args.train_from_scratch and not args.train_auto_inherit and reference_joints != target_joints:
                raise ValueError('PPO inheritance requires identical ordered joint names; use --train_from_scratch for changed topology')
            inherit_weights = not args.train_from_scratch
            if args.train_auto_inherit and saved_policy_contract.get('controller') != controller:
                raise ValueError('Inherited policy controller differs from child controller')
            from mechanical_ppo import train_mechanical
            train_mechanical(raw_env, transfer, resume_path, output_dir, args.train_iterations,
                             args.train_smoke, inherit_weights=inherit_weights,
                             bc_manifest=args.bc_manifest,bc_only=args.bc_only,task=args.task)
            return
        if args.replay_policy_trace:
            raise ValueError('Mechanical replay requires a separately converted trace contract')
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
                        actions = (torch.as_tensor(replay_actions[step], device=raw_env.unwrapped.device).unsqueeze(0) if replay_actions is not None and (args.replay_until_step == 0 or step < args.replay_until_step) else transfer.action_for_target_hand(agent.get_action(agent.obs_to_torch(transfer.observation_for_source_policy(obs)), is_deterministic=True)))
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
                            physical = raw_env.unwrapped
                            actual_actions = physical.raw_actions if hasattr(physical, "raw_actions") else physical.actions
                            executed_actions.append(actual_actions[0].detach().cpu().numpy().astype("float32"))
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
            "forage_solver_iterations_diagnostic": args.forage_solver_iterations,
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
        print('[MECHANICAL_EVAL_RESULT]', json.dumps({'task': args.task, 'successes': successes, 'report': str(output_dir / 'evaluation.json')}), flush=True)
        if _worker_mode:
            return
        os._exit(0)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        # This is a dedicated per-episode process. Kit teardown can deadlock
        # and swallow the original exception; preserve the real exit status.
        if _worker_mode:
            raise
        os._exit(1)


if __name__ == "__main__":
    main()
