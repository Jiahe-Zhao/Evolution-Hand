import gymnasium as gym

from isaaclab_tasks.evolution_tasks.task_grasp import agents


gym.register(
    id="Isaac-EvolutionHand-Forage-v0",
    entry_point="isaaclab_tasks.evolution_tasks.task_forage.forage_env:ForageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg:ForageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
