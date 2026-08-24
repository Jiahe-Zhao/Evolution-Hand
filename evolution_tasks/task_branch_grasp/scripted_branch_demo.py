"""Run a deterministic BranchGrasp closing trajectory and save an MP4 demonstration.

The action profile is deliberately expressed in the task's normalized [-1, 1]
joint-action space.  It is intended as a reproducible seed trajectory for
later behavior-cloning experiments, not as a learned policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record a scripted BranchGrasp demonstration.")
parser.add_argument("--output", type=str, required=True, help="Output MP4 path.")
parser.add_argument("--metrics", type=str, required=True, help="Output JSON metrics path.")
parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--hold_steps", type=int, default=90)
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
from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg


def frame_to_uint8(frame):
    frame = frame[0] if isinstance(frame, (list, tuple)) else frame
    frame = np.asarray(frame)
    return frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)


def branch_action(step: int, device: torch.device) -> torch.Tensor:
    """Open, settle, then close the thumb and four opposing fingers together."""
    if step < 30:
        value = -0.70
    elif step < 90:
        value = -0.70 + 1.45 * (step - 30) / 60.0
    else:
        value = 0.90

    # Finger 1 is the task's thumb contact channel.  The remaining four
    # fingers oppose it, so the same gradual flexion profile forms a pinch.
    return torch.full((1, 19), value, dtype=torch.float32, device=device)


def main() -> None:
    os.environ["EVOLUTION_CURRICULUM_STAGE"] = args_cli.stage
    cfg = BranchGraspEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.reset_dof_pos_noise = 0.0
    cfg.seed = args_cli.seed
    # The branch is placed in a reproducible pre-grasp state between the
    # thumb and opposing fingertips.  The task reward and success test stay
    # unchanged; this is only the demonstration/reset state.
    cfg.branch_cfg.init_state.pos = (0.0, 0.012, 0.30)
    if hasattr(cfg, "viewer"):
        cfg.viewer.eye = (0.24, -0.18, 0.36)
        cfg.viewer.lookat = (0.0, 0.0, 0.30)

    env = gym.make("Isaac-EvolutionHand-BranchGrasp-v0", cfg=cfg, render_mode="rgb_array")
    env.reset(seed=args_cli.seed)
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[0.24, -0.18, 0.36], target=[0.0, 0.0, 0.30], camera_prim_path="/OmniverseKit_Persp"
        )
    except Exception as error:
        print(f"camera_view_warning={error}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    writer = imageio.get_writer(args_cli.output, fps=30, codec="libx264", quality=8)
    history: list[dict[str, float | int | bool]] = []
    success = False
    total_steps = 90 + args_cli.hold_steps
    try:
        for step in range(total_steps):
            action = branch_action(step, env.unwrapped.device)
            _, reward, terminated, truncated, _ = env.step(action)
            unwrapped = env.unwrapped
            forces = torch.norm(unwrapped.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1)[0]
            thumb_force = float(forces[0].item())
            other_force = float(forces[1:].max().item())
            hold_steps = int(unwrapped.branch_success_streak[0].item())
            reward_value = float(reward[0].item())
            success = success or reward_value >= unwrapped.cfg.success_reward
            history.append(
                {
                    "step": step,
                    "thumb_force_n": thumb_force,
                    "other_finger_force_n": other_force,
                    "hold_steps": hold_steps,
                    "reward": reward_value,
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                }
            )
            frame = env.render()
            if frame is not None:
                writer.append_data(frame_to_uint8(frame))
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                break
    finally:
        writer.close()

    summary = {
        "task": "BranchGrasp",
        "stage": args_cli.stage,
        "success": success,
        "required_force_n": cfg.branch_contact_force_threshold,
        "required_hold_steps": cfg.branch_success_hold_steps,
        "max_thumb_force_n": max(item["thumb_force_n"] for item in history),
        "max_other_finger_force_n": max(item["other_finger_force_n"] for item in history),
        "max_hold_steps": max(item["hold_steps"] for item in history),
        "steps_executed": len(history),
        "trajectory": "open(30)->linear_close(60)->hold",
        "history": history,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics)), exist_ok=True)
    with open(args_cli.metrics, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
