import gymnasium as gym

from isaaclab_tasks.evolution_tasks.task_grasp import agents


gym.register(
    id="Isaac-EvolutionHand-Carry-v0",
    entry_point="isaaclab_tasks.evolution_tasks.task_carry.carry_env:CarryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.evolution_tasks.task_carry.carry_env_cfg:CarryEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
