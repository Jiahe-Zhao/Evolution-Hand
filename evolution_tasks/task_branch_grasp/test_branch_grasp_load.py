import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_branch_grasp  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg import BranchGraspEnvCfg


def main():
    print("phase=imports_ok", flush=True)
    cfg = BranchGraspEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    print("phase=cfg_ok", cfg.action_space, cfg.observation_space, flush=True)
    env = gym.make("Isaac-EvolutionHand-BranchGrasp-v0", cfg=cfg)
    print("phase=make_ok", flush=True)

    obs, _ = env.reset()
    print("reset_ok", list(obs.keys()), obs["policy"].shape, flush=True)

    actions = torch.zeros((cfg.scene.num_envs, cfg.action_space), device=env.unwrapped.device)
    obs, reward, terminated, truncated, _ = env.step(actions)
    print("step_ok", obs["policy"].shape, reward.shape, terminated.shape, truncated.shape, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
