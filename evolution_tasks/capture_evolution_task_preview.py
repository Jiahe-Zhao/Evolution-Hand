import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture one preview image for an evolution task.")
parser.add_argument("--task-key", type=str, required=True, help="Task key: branch|carry|forage|grasp|manipulation|strike|stone")
parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=16, help="Number of simulation steps before saving the frame.")
parser.add_argument(
    "--action-value",
    type=float,
    default=0.0,
    help="Constant normalized joint command in [-1, 1]; positive values close the hand.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_branch_grasp  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_carry  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_forage  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_grasp  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_manipulation  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_strike  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_stone  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg
from isaaclab_tasks.evolution_tasks.task_carry.carry_env_cfg import CarryEnvCfg
from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg
from isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg import EvolutionGraspEnvCfg
from isaaclab_tasks.evolution_tasks.task_manipulation.evolution_manipulation_env_cfg import EvolutionManipulationEnvCfg
from isaaclab_tasks.evolution_tasks.task_stone.evolution_stone_grind_env_cfg import EvolutionStoneGrindEnvCfg
from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg


TASK_SPECS = {
    "carry": {
        "task_id": "Isaac-EvolutionHand-Carry-v0",
        "cfg_cls": CarryEnvCfg,
        "camera_eye": (0.18, -0.22, 0.62),
        "camera_target": (0.01, 0.0, 0.36),
    },
    "branch": {
        "task_id": "Isaac-EvolutionHand-BranchGrasp-v0",
        "cfg_cls": BranchGraspEnvCfg,
        "camera_eye": (0.24, -0.16, 0.37),
        "camera_target": (0.0, 0.0, 0.315),
    },
    "forage": {
        "task_id": "Isaac-EvolutionHand-Forage-v0",
        "cfg_cls": ForageEnvCfg,
        "camera_eye": (0.28, -0.26, 0.42),
        "camera_target": (0.0, 0.0, 0.11),
    },
    "grasp": {
        "task_id": "Isaac-EvolutionHand-Grasp-v0",
        "cfg_cls": EvolutionGraspEnvCfg,
        # A near-top view makes the continuous palmar surface visible instead
        # of collapsing it into a thin edge-on silhouette.
        "camera_eye": (0.02, -0.02, 0.68),
        "camera_target": (-0.02, 0.00, 0.35),
    },
    "manipulation": {
        "task_id": "Isaac-EvolutionHand-Manipulation-v0",
        "cfg_cls": EvolutionManipulationEnvCfg,
        "camera_eye": (0.22, -0.18, 0.46),
        "camera_target": (-0.03, 0.00, 0.36),
    },
    "strike": {
        "task_id": "Isaac-EvolutionHand-Strike-v0",
        "cfg_cls": EvolutionStrikeEnvCfg,
        "camera_eye": (0.20, -0.16, 0.34),
        "camera_target": (-0.03, 0.01, 0.18),
    },
    "stone": {
        "task_id": "Isaac-EvolutionHand-StoneGrind-v0",
        "cfg_cls": EvolutionStoneGrindEnvCfg,
        "camera_eye": (0.04, -0.04, 0.70),
        "camera_target": (-0.03, 0.01, 0.39),
    },
}


def _to_uint8(frame):
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _make_zero_action(env_unwrapped):
    if hasattr(env_unwrapped.cfg, "action_space"):
        action_dim = env_unwrapped.cfg.action_space
        return torch.full(
            (env_unwrapped.num_envs, action_dim),
            args_cli.action_value,
            dtype=torch.float32,
            device=env_unwrapped.device,
        )
    if hasattr(env_unwrapped.cfg, "action_spaces"):
        actions = {}
        for agent_name, action_dim in env_unwrapped.cfg.action_spaces.items():
            actions[agent_name] = torch.full(
                (env_unwrapped.num_envs, action_dim),
                args_cli.action_value,
                dtype=torch.float32,
                device=env_unwrapped.device,
            )
        return actions
    raise RuntimeError("Unsupported action space format")


def _set_camera(eye, target):
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=list(eye),
            target=list(target),
            camera_prim_path="/OmniverseKit_Persp",
        )
        print(f"camera_view_set_ok eye={eye} target={target}")
    except Exception as error:
        print(f"camera_view_set_failed {error}")


def _apply_palmar_membrane_material():
    """The membrane is a porous shell so it stays legible across URDF renderers."""
    print("palmar_membrane_visual=porous_shell")


def _print_tracked_object_state(env_unwrapped):
    for asset_name in ("grasp_object", "object", "cone", "Cube"):
        try:
            asset = env_unwrapped.scene[asset_name]
        except Exception:
            continue
        data = getattr(asset, "data", None)
        root_pos = getattr(data, "root_pos_w", None)
        root_quat = getattr(data, "root_quat_w", None)
        if root_pos is None:
            continue
        pos = root_pos[0].detach().cpu().tolist()
        quat = root_quat[0].detach().cpu().tolist() if root_quat is not None else None
        print(f"tracked_object_state name={asset_name} pos={pos} quat={quat}")
        return
    print("tracked_object_state unavailable")


def main():
    spec = TASK_SPECS[args_cli.task_key]
    cfg = spec["cfg_cls"]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 2.0
    if hasattr(cfg, "viewer"):
        cfg.viewer.eye = spec["camera_eye"]
        cfg.viewer.lookat = spec["camera_target"]

    env = gym.make(spec["task_id"], cfg=cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    print(f"reset_ok task={args_cli.task_key} obs_type={type(obs).__name__}")

    _apply_palmar_membrane_material()
    _set_camera(spec["camera_eye"], spec["camera_target"])
    actions = _make_zero_action(env.unwrapped)

    frame = None
    for step in range(args_cli.steps):
        env.step(actions)
        frame = env.render()
        if frame is not None:
            arr = np.asarray(frame[0] if isinstance(frame, (list, tuple)) else frame)
            print(f"frame_step={step} shape={arr.shape} dtype={arr.dtype}")

    if frame is None:
        raise RuntimeError("env.render() returned None")

    frame = _to_uint8(frame)
    _print_tracked_object_state(env.unwrapped)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    imageio.imwrite(args_cli.output, frame)
    print(f"saved={args_cli.output}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
