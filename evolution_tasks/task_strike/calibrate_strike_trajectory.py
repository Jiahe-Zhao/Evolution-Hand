from __future__ import annotations
import argparse, json, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_strike  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg

profiles = [1.0, -1.0, 0.8, -0.8]
cfg = EvolutionStrikeEnvCfg(); cfg.scene.num_envs = len(profiles); cfg.reset_dof_pos_noise = 0.0; cfg.seed = 7
# Reproducible pre-impact state: align the fixed wrist with the cone centre.
cfg.robot_cfg.init_state.pos = (-0.05, 0.01, 0.275)
env = gym.make("Isaac-EvolutionHand-Strike-v0", cfg=cfg); env.reset(seed=7)
results = [{"terminal": value, "max_force_n": 0.0, "max_reward": 0.0} for value in profiles]
for step in range(100):
    actions = torch.full((len(profiles), 19), -1.0 if step < 25 else 0.0, device=env.unwrapped.device)
    if step >= 40:
        for index, value in enumerate(profiles): actions[index] = value
    _, rewards, _, _, _ = env.step(actions); env.unwrapped._compute_intermediate_values()
    for index, result in enumerate(results):
        result["max_force_n"] = max(result["max_force_n"], float(torch.norm(env.unwrapped.strike_object_force[index])))
        result["max_reward"] = max(result["max_reward"], float(rewards[index]))
json.dump(results, open(args.output, "w"), indent=2); print(results)
env.close(); app.close()
