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

cfg = EvolutionStrikeEnvCfg()
cfg.scene.num_envs = 1
cfg.reset_dof_pos_noise = 0.0
cfg.seed = 7
cfg.robot_cfg.init_state.pos = (-0.05, 0.01, 0.275)
env = gym.make("Isaac-EvolutionHand-Strike-v0", cfg=cfg, render_mode="rgb_array")
env.reset(seed=7)

# A wider elevated view includes the whole hand, cone tip, and target block.
set_camera_view(eye=np.array([-0.46, -0.40, 0.46]), target=np.array([-0.05, 0.01, 0.19]))
os.makedirs(os.path.dirname(args.output), exist_ok=True)
writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
history = []
for step in range(100):
    action = torch.full((1, 19), -1.0 if step < 25 else 0.0, device=env.unwrapped.device)
    if step >= 40:
        action.fill_(-1.0)
    _, reward, _, _, _ = env.step(action)
    env.unwrapped._compute_intermediate_values()
    force = float(torch.norm(env.unwrapped.strike_object_force[0]))
    tip = env.unwrapped.cone_tip_pos[0]
    distance = float(torch.norm(tip[:2] - env.unwrapped.strike_target_pos[0, :2]))
    history.append({"step": step, "force_n": force, "tip_goal_distance_m": distance, "reward": float(reward[0])})
    frame = env.render()
    if frame is not None:
        writer.append_data(np.asarray(frame[0] if isinstance(frame, (tuple, list)) else frame).astype(np.uint8))
    if reward[0] >= 1000:
        break
writer.close()
with open(args.metrics, "w") as file:
    json.dump({"task": "Strike", "success": any(x["reward"] >= 1000 for x in history), "max_force_n": max(x["force_n"] for x in history), "steps_executed": len(history), "history": history}, file, indent=2)
env.close()
app.close()
