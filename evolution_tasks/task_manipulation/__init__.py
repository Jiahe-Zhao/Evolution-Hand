# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Shadow Hand environment.
"""

import gymnasium as gym

from . import agents  # 就是指向同路径的agents文件夹

##
# Register Gym environments.
##

""""""


gym.register(
    id="Isaac-ShadowHand-Manipulation-v0", # 任务名称要保持唯一
    entry_point="isaaclab_tasks.evolution_tasks.task_manipulation.shadowhand_manipulation_env:ManipulationEnv",  # 定义任务（包括任务目标，奖励啥的，重置啥的）
    disable_env_checker=True, # 禁用环境检查器
    kwargs={
        # 定义对象
        "env_cfg_entry_point": "isaaclab_tasks.evolution_tasks.task_manipulation.shadowhand_manipulation_env_cfg:ShadowHandEnvCfg",
        # 提供rl_games 库的PPO（近端策略优化）算法的配置
        # 包括胜利条件，迭代Isaac-ShadowHand-StoneGrind-v0次数，多少次储存，batchsize等
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # rl的PPO算法的参数配置，这个好像是对不同观察方式（如视觉等）来讲
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        # 这段代码配置了一个基于 skrl 库的强化学习项目，使用了PPO（Proximal Policy Optimization）算法，
        # 并应用了 GaussianMixin 和 DeterministicMixin 模型。
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",

    },
)

gym.register(
    id="Isaac-EvolutionHand-Manipulation-v0", # 任务名称要保持唯一
    entry_point="isaaclab_tasks.evolution_tasks.task_manipulation.evolution_manipulation_env:EvolutionManipulationEnv",  # 定义任务（包括任务目标，奖励啥的，重置啥的）（否）判断是否需要在——isaaclab_other中随着agent更新
    disable_env_checker=True, # 禁用环境检查器
    kwargs={
        # 定义对象
        "env_cfg_entry_point": "isaaclab_tasks.evolution_tasks.task_manipulation.evolution_manipulation_env_cfg:EvolutionManipulationEnvCfg",# 需要根据agent更新
        # 提供rl_games 库的PPO（近端策略优化）算法的配置
        # 包括胜利条件，迭代Isaac-ShadowHand-StoneGrind-v0次数，多少次储存，batchsize等
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # rl的PPO算法的参数配置，这个好像是对不同观察方式（如视觉等）来讲
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ShadowHandPPORunnerCfg",
        # 这段代码配置了一个基于 skrl 库的强化学习项目，使用了PPO（Proximal Policy Optimization）算法，
        # 并应用了 GaussianMixin 和 DeterministicMixin 模型。
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",

    },
)

# ./isaaclab.sh -p /home/p/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/train_interface.py --num_envs=64 --task=Isaac-Hand-Cube-Evolution-v0