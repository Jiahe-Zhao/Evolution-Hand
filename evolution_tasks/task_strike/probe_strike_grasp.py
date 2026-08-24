from __future__ import annotations

from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True, "enable_cameras": False}).app

import gymnasium as gym
import torch

import isaaclab_tasks
import isaaclab_tasks.evolution_tasks.task_strike
from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg

cfg = EvolutionStrikeEnvCfg()
cfg.scene.num_envs = 1
cfg.reset_dof_pos_noise = 0.0
env = gym.make("Isaac-EvolutionHand-Strike-v0", cfg=cfg)
env.reset(seed=7)
for _ in range(5):
    env.step(torch.zeros((1, 19), device=env.unwrapped.device))
env.unwrapped._compute_intermediate_values()
print("root", env.unwrapped.hand.data.root_pos_w[0].detach().cpu().tolist())
print("cone", env.unwrapped.cone_pos[0].detach().cpu().tolist())
for name, pos in zip(cfg.fingertip_body_names, env.unwrapped.fingertip_pos[0].detach().cpu().tolist()):
    print(name, pos)
env.close()
app.close()
