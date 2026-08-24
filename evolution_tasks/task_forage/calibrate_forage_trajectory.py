"""Measure deterministic fingertip sweeps and leaf-clearance outcomes for Forage."""
from __future__ import annotations
import argparse, json, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_forage  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg

PROFILES = [(0.9, 0.9), (-0.9, 0.9), (0.9, -0.9), (-0.9, -0.9), (0.0, 0.9), (0.9, 0.0)]

def actions(step, device):
    # Phase one flexes toward leaf one.  Phase two reverses the four fingers,
    # producing a second physical sweep rather than teleporting either leaf.
    alpha = min(1.0, max(0.0, (step - 20) / 55.0))
    phase_two = step >= 105
    result = torch.full((len(PROFILES), 19), -0.6, device=device)
    for index, (thumb, fingers) in enumerate(PROFILES):
        result[index, :3] = -0.6 + alpha * (thumb + 0.6)
        terminal_fingers = -0.9 if phase_two else fingers
        result[index, 3:] = -0.6 + alpha * (terminal_fingers + 0.6)
    return result

def main():
    cfg = ForageEnvCfg(); cfg.scene.num_envs = len(PROFILES); cfg.reset_dof_pos_noise = 0.0; cfg.seed = 7
    # First leaf starts just inside the 7.5 cm clearance threshold, so the
    # physical sweep can establish the required ordered first milestone.
    cfg.leaf_one_cfg.init_state.pos = (0.074, 0.0, 0.063)
    cfg.leaf_two_cfg.init_state.pos = (0.050, 0.002, 0.068)
    env = gym.make("Isaac-EvolutionHand-Forage-v0", cfg=cfg); env.reset(seed=7)
    result = [{"thumb_final": p[0], "other_final": p[1], "max_reward": 0.0,
               "cumulative_reward": 0.0, "success": False} for p in PROFILES]
    for step in range(180):
        _, rewards, terminated, _, _ = env.step(actions(step, env.unwrapped.device))
        for index, item in enumerate(result):
            reward = float(rewards[index])
            item["max_reward"] = max(item["max_reward"], reward)
            item["cumulative_reward"] += reward
            item["success"] = item["success"] or (reward >= 700.0 and bool(terminated[index]))
    env.unwrapped._compute_intermediate_values()
    for index, item in enumerate(result):
        item["tip_positions_m"] = env.unwrapped.fingertip_pos[index].detach().cpu().tolist()
        item["leaf_one_pos_m"] = env.unwrapped.leaf_one_pos[index].detach().cpu().tolist()
        item["leaf_two_pos_m"] = env.unwrapped.leaf_two_pos[index].detach().cpu().tolist()
        item["food_pos_m"] = env.unwrapped.food_pos[index].detach().cpu().tolist()
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    json.dump(result, open(args_cli.output, "w"), indent=2); print(json.dumps(result, indent=2))
    env.close(); simulation_app.close()
if __name__ == "__main__": main()
