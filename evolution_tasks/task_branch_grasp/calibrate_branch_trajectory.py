"""Parallel action-sign sweep for the scripted BranchGrasp demonstration."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Find a deterministic BranchGrasp pinch action profile.")
parser.add_argument("--output", required=True, help="JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_branch_grasp  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg


# (thumb terminal action, terminal action for fingers 2--5).  The first three
# dimensions are the sensor-defined thumb; remaining dimensions are opponents.
PROFILES = [(0.90, 0.90), (-0.90, 0.90), (0.90, -0.90), (-0.90, -0.90), (0.0, 0.90), (0.90, 0.0)]


def actions(step: int, device: torch.device) -> torch.Tensor:
    alpha = min(1.0, max(0.0, (step - 20) / 70.0))
    result = torch.full((len(PROFILES), 19), -0.60, dtype=torch.float32, device=device)
    for index, (thumb_final, fingers_final) in enumerate(PROFILES):
        result[index, :3] = -0.60 + alpha * (thumb_final + 0.60)
        result[index, 3:] = -0.60 + alpha * (fingers_final + 0.60)
    return result


def main() -> None:
    os.environ["EVOLUTION_CURRICULUM_STAGE"] = "stage1"
    cfg = BranchGraspEnvCfg()
    cfg.scene.num_envs = len(PROFILES)
    cfg.reset_dof_pos_noise = 0.0
    cfg.seed = 7
    # Pre-grasp curriculum state: place the fixed branch between the thumb
    # and opposing fingertips without changing its orientation or reward.
    cfg.branch_cfg.init_state.pos = (0.0, 0.012, 0.30)
    env = gym.make("Isaac-EvolutionHand-BranchGrasp-v0", cfg=cfg)
    env.reset(seed=7)
    result = [
        {"thumb_final": profile[0], "other_final": profile[1], "max_thumb_force_n": 0.0,
         "max_other_force_n": 0.0, "max_hold_steps": 0, "success": False}
        for profile in PROFILES
    ]
    for step in range(180):
        _, reward, _, _, _ = env.step(actions(step, env.unwrapped.device))
        forces = torch.norm(env.unwrapped.branch_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=-1)
        holds = env.unwrapped.branch_success_streak
        for index, item in enumerate(result):
            item["max_thumb_force_n"] = max(item["max_thumb_force_n"], float(forces[index, 0]))
            item["max_other_force_n"] = max(item["max_other_force_n"], float(forces[index, 1:].max()))
            item["max_hold_steps"] = max(item["max_hold_steps"], int(holds[index]))
            item["success"] = item["success"] or bool(reward[index] >= cfg.success_reward)
    env.unwrapped._compute_intermediate_values()
    tip_positions = env.unwrapped.fingertip_pos.detach().cpu().tolist()
    branch_positions = env.unwrapped.branch_pos.detach().cpu().tolist()
    for index, item in enumerate(result):
        item["final_tip_positions_m"] = tip_positions[index]
        item["branch_position_m"] = branch_positions[index]
    with open(args_cli.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
