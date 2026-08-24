"""Record a complete Forage episode with an observable idle and hold period."""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--metrics", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks
import isaaclab_tasks.evolution_tasks.task_forage
from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg

IDLE_STEPS = 45
POST_SUCCESS_FRAMES = 60


def scripted_action(step: int, device: torch.device) -> torch.Tensor:
    """Keep the hand neutral first, then replay the existing gradual sweep."""
    if step < IDLE_STEPS:
        return torch.zeros((1, 19), dtype=torch.float32, device=device)
    local_step = step - IDLE_STEPS
    alpha = min(1.0, max(0.0, (local_step - 20) / 55.0))
    result = torch.full((1, 19), -0.6, dtype=torch.float32, device=device)
    result[:, :3] = -0.6 + alpha * 1.5
    result[:, 3:] = -0.6 + alpha * ((-0.3 if local_step >= 105 else 1.5))
    return result


def frame_u8(frame):
    frame = frame[0] if isinstance(frame, (tuple, list)) else frame
    return np.asarray(frame).astype(np.uint8)


cfg = ForageEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.0
cfg.reset_dof_pos_noise = 0.0
cfg.seed = 7
env = gym.make("Isaac-EvolutionHand-Forage-v0", cfg=cfg, render_mode="rgb_array")
env.reset(seed=7)

from isaacsim.core.utils.viewports import set_camera_view
set_camera_view(eye=[0.28, -0.26, 0.42], target=[0.0, 0.0, 0.11], camera_prim_path="/OmniverseKit_Persp")
os.makedirs(os.path.dirname(args.output), exist_ok=True)
writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
history = []
success_step = None
try:
    for step in range(240):
        _, reward, terminated, truncated, _ = env.step(scripted_action(step, env.unwrapped.device))
        env.unwrapped._compute_intermediate_values()
        food = env.unwrapped.food_pos[0, :2]
        d1 = float(torch.norm(env.unwrapped.leaf_one_pos[0, :2] - food))
        d2 = float(torch.norm(env.unwrapped.leaf_two_pos[0, :2] - food))
        value = float(reward[0])
        history.append({"step": step, "reward": value, "leaf_one_distance_m": d1, "leaf_two_distance_m": d2})
        writer.append_data(frame_u8(env.render()))
        if value >= 700.0:
            success_step = step
            break
        if terminated[0] or truncated[0]:
            break
    # Do not step a terminated environment: hold its final rendered state instead.
    for _ in range(POST_SUCCESS_FRAMES):
        writer.append_data(frame_u8(env.render()))
finally:
    writer.close()

with open(args.metrics, "w") as file:
    json.dump({"task": "Forage", "idle_steps": IDLE_STEPS, "success_step": success_step, "steps_executed": len(history), "history": history}, file, indent=2)
env.close()
app.close()
