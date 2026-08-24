"""Record a complete Strike episode with an observable idle, impact, and hold."""
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
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab_tasks
import isaaclab_tasks.evolution_tasks.task_strike
from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg

IDLE_STEPS = 45
RAMP_STEPS = 35
POST_SUCCESS_FRAMES = 60


def strike_action(step: int, device: torch.device) -> torch.Tensor:
    # Keep the trained pre-grasp unchanged, then use only the Cartesian wrist
    # z channel to drive the held tool toward the impact block.
    action = torch.zeros((1, 22), dtype=torch.float32, device=device)
    if step >= IDLE_STEPS:
        action[:, 21] = -min(1.0, (step - IDLE_STEPS + 1) / RAMP_STEPS)
    return action


def frame_u8(frame):
    frame = frame[0] if isinstance(frame, (tuple, list)) else frame
    return np.asarray(frame).astype(np.uint8)


cfg = EvolutionStrikeEnvCfg()
cfg.scene.num_envs = 1
cfg.reset_dof_pos_noise = 0.0
cfg.seed = 7
cfg.robot_cfg.init_state.pos = (-0.05, 0.01, 0.380)
env = gym.make("Isaac-EvolutionHand-Strike-v0", cfg=cfg, render_mode="rgb_array")
env.reset(seed=7)
set_camera_view(eye=np.array([-0.46, -0.40, 0.46]), target=np.array([-0.05, 0.01, 0.19]))
os.makedirs(os.path.dirname(args.output), exist_ok=True)
writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
history = []
success_step = None
try:
    for step in range(180):
        _, reward, terminated, truncated, _ = env.step(strike_action(step, env.unwrapped.device))
        env.unwrapped._compute_intermediate_values()
        force = float(torch.norm(env.unwrapped.strike_object_force[0]))
        tip = env.unwrapped.cone_tip_pos[0]
        distance = float(torch.norm(tip[:2] - env.unwrapped.strike_target_pos[0, :2]))
        value = float(reward[0])
        history.append(
            {
                "step": step,
                "reward": value,
                "force_n": force,
                "tip_goal_distance_m": distance,
                "cone_root_height_m": float(env.unwrapped.cone_pos[0, 2]),
                "tool_was_held": bool(env.unwrapped.tool_was_held[0]),
            }
        )
        writer.append_data(frame_u8(env.render()))
        if value >= 1000.0:
            success_step = step
            break
        if terminated[0] or truncated[0]:
            break
    for _ in range(POST_SUCCESS_FRAMES):
        writer.append_data(frame_u8(env.render()))
finally:
    writer.close()

with open(args.metrics, "w") as file:
    json.dump({"task": "Strike", "idle_steps": IDLE_STEPS, "success_step": success_step, "steps_executed": len(history), "history": history}, file, indent=2)
env.close()
app.close()
