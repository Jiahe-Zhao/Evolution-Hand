"""Search reset placements that are physically retained by the closed hand."""
from __future__ import annotations

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import isaaclab_tasks
import isaaclab_tasks.evolution_tasks.task_strike
from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg


cfg = EvolutionStrikeEnvCfg()
cfg.scene.num_envs = 1
cfg.reset_dof_pos_noise = 0.0
cfg.seed = 7
env = gym.make("Isaac-EvolutionHand-Strike-v0", cfg=cfg)
for x in (-0.035, -0.045, -0.055, -0.065, -0.075, -0.085):
    env.reset(seed=7)
    state = env.unwrapped.cone.data.default_root_state.clone()
    state[:, :3] = torch.tensor((x, 0.01, 0.275), device=env.unwrapped.device)
    state[:, 7:] = 0.0
    env.unwrapped.cone.write_root_state_to_sim(state)
    action = torch.zeros((1, 22), device=env.unwrapped.device)
    for _ in range(60):
        env.step(action)
    env.unwrapped._compute_intermediate_values()
    pos = env.unwrapped.cone_pos[0].tolist()
    print(f"RETENTION x={x:.3f} final=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
env.close()
app.close()
