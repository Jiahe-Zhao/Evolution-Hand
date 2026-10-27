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
sys.path.insert(0,'/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021/four_tasks')
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
parser.add_argument('--finite_check',action='store_true')
parser.add_argument('--benchmark',action='store_true')
parser.add_argument('--steps',type=int,default=32768)
parser.add_argument('--horizon',type=int,default=128)
parser.add_argument('--eval_seed',type=int,default=1062701)
parser.add_argument('--eval_every',type=int,default=1048576)
parser.add_argument('--baseline_genotype',required=True)
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
    args.individual_key='baseline'
    code_root = Path(os.environ.get("EVOLUTION_CODE_ROOT", "/home/zjh/Evolution_PC"))
    sys.path.insert(0, str(code_root / "Isaaclab_other"))
    from code_to_urdf import generate_urdf_from_dict
    from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
    from mirror_agent import create_mirror_hand

    hand = json.loads(Path(args.baseline_genotype).read_text())
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
    curriculum_stage = 'stage2'
    os.environ["EVOLUTION_CURRICULUM_STAGE"] = curriculum_stage
    os.environ["EVOLUTION_FORAGE_CURRICULUM_STAGE"] = curriculum_stage
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
        from native_contact_hooks import install as install_baseline
        install_baseline(env_id,env_cfg,output_dir,args.task)
        raw_env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if args.record_video else None)
        raw = raw_env.unwrapped
        obs, _ = raw_env.reset(seed=args.seed)
        if args.finite_check:
            from limited_native_check import run as check
            check(raw_env,graph,args,output_dir)
            return
        from native_goal_adapter import GoalAdapter
        from native_goal_ppo import run
        wrapped=GoalAdapter(raw_env,args.task,output_dir)
        args.episode_steps=raw.max_episode_length
        args.smoke=args.benchmark
        args.source=str(mechanical_urdf)
        run(wrapped,args,output_dir)
    finally:
        if 'raw_env' in locals(): raw_env.close()

if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
