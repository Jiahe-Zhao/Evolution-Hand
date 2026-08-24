"""Task identifiers and configuration entry points shared by suite scripts."""

TASKS = {
    "grasp": (
        "Isaac-EvolutionHand-Grasp-v0",
        "isaaclab_tasks.evolution_tasks.task_grasp",
        "isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg",
        "EvolutionGraspEnvCfg",
    ),
    "branch": (
        "Isaac-EvolutionHand-BranchGrasp-v0",
        "isaaclab_tasks.evolution_tasks.task_branch_grasp",
        "isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg",
        "BranchGraspEnvCfg",
    ),
    "forage": (
        "Isaac-EvolutionHand-Forage-v0",
        "isaaclab_tasks.evolution_tasks.task_forage",
        "isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg",
        "ForageEnvCfg",
    ),
    "strike": (
        "Isaac-EvolutionHand-Strike-v0",
        "isaaclab_tasks.evolution_tasks.task_strike",
        "isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg",
        "EvolutionStrikeEnvCfg",
    ),
}
