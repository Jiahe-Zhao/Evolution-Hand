import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Capture one frame from BranchGrasp task.")
parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
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


def _to_uint8(frame):
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main():
    cfg = BranchGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 2.0
    if hasattr(cfg, "viewer"):
        cfg.viewer.eye = (0.24, -0.16, 0.37)
        cfg.viewer.lookat = (0.0, 0.0, 0.315)

    env = gym.make("Isaac-EvolutionHand-BranchGrasp-v0", cfg=cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    print(f"reset_ok {list(obs.keys())}")

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[0.24, -0.16, 0.37],
            target=[0.0, 0.0, 0.315],
            camera_prim_path="/OmniverseKit_Persp",
        )
        print("camera_view_set_ok")
    except Exception as error:
        print(f"camera_view_set_failed {error}")

    action_dim = env.unwrapped.cfg.action_space
    zeros = torch.zeros((env.unwrapped.num_envs, action_dim), dtype=torch.float32, device=env.unwrapped.device)

    frame = None
    for step in range(12):
        env.step(zeros)
        frame = env.render()
        if frame is not None:
            arr = np.asarray(frame[0] if isinstance(frame, (list, tuple)) else frame)
            print(f"frame_step={step} shape={arr.shape} dtype={arr.dtype}")

    if frame is None:
        raise RuntimeError("env.render() returned None")

    frame = _to_uint8(frame)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    imageio.imwrite(args_cli.output, frame)
    print(f"saved={args_cli.output}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
