#!/bin/bash

set -euo pipefail

EVOLUTION_ROOT=/home/zjh/Evolution_PC
ISAACLAB_ROOT=/home/zjh/IsaacLab
ISAAC_ENV_PREFIX=/home/zjh/miniconda3/envs/evolution_isaaclab

export TERM=xterm PYTHONUNBUFFERED=1 ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export CUDA_VISIBLE_DEVICES=0
export CONDA_PREFIX="$ISAAC_ENV_PREFIX" PATH="$ISAAC_ENV_PREFIX/bin:$PATH"
export EVOLUTION_ROOT ISAACLAB_ROOT ISAAC_SIM_SETUP=/dev/null
export EVOLUTION_LOG_ROOT="$EVOLUTION_ROOT/evolution_tasks/logs"

# Two slots split the original 4096 environments into 2048 each.  This keeps
# two individuals in flight without placing two independent 4096-env scenes
# on the same RTX 4090.
export ISAACLAB_NUM_ENVS=4096
export EVOLUTION_PARALLEL_SLOTS=2
export EVOLUTION_PARALLEL_SPLIT_ENVS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

export EVOLUTION_PPO_HORIZON_LENGTH=16
export EVOLUTION_PPO_MINIBATCH_SIZE=4096
export EVOLUTION_PPO_MINI_EPOCHS=5
export ISAACLAB_MAX_ITERATIONS=800
export EVOLUTION_STAGE1_MAX_ITERATIONS=200
export EVOLUTION_STAGE2_MAX_ITERATIONS=800
export EVOLUTION_STAGE2_TOP_FRACTION=1.0

# Grasp and Branch retain the full 200 -> 800 curriculum.  Forage/Strike are
# already saturated, so every morphology still receives a compact validation
# run without spending the full budget on tasks that no longer rank designs.
export EVOLUTION_TASK_MAX_ITERATIONS_STAGE1='Isaac-EvolutionHand-Forage-v0=50,Isaac-EvolutionHand-Strike-v0=50'
export EVOLUTION_TASK_MAX_ITERATIONS_STAGE2='Isaac-EvolutionHand-Forage-v0=100,Isaac-EvolutionHand-Strike-v0=100'

# Reuse once to amortize Isaac startup, then recycle before repeated native
# scene switches accumulate state across a long job.
export EVOLUTION_REUSE_ISAAC_PROCESS=1
export EVOLUTION_ISAAC_WORKER_MAX_REQUESTS=2
export EVOLUTION_ISAAC_WORKER_STALL_TIMEOUT=600
export EVOLUTION_ISAAC_WORKER_REQUEST_TIMEOUT=14400
export EVOLUTION_CHECKPOINT_INTERVAL=50
export EVOLUTION_KEEP_LATEST_CHECKPOINTS=1 EVOLUTION_KEEP_BEST_CHECKPOINTS=1

export EVOLUTION_EXPERIMENT_NAME="${EVOLUTION_EXPERIMENT_NAME:-exp_20260827_fast_4tasks_grasp_m3thumb2_env2048x2}"
export EVOLUTION_FORCE_NEW_LINEAGE="${EVOLUTION_FORCE_NEW_LINEAGE:-1}"
export EVOLUTION_MAX_GENERATION="${EVOLUTION_MAX_GENERATION:-15}"
export EVOLUTION_MAX_POPULATION="${EVOLUTION_MAX_POPULATION:-16}"
export EVOLUTION_MAX_VARIATION="${EVOLUTION_MAX_VARIATION:-1}"
export EVOLUTION_INITIAL_POPULATION_SIZE="${EVOLUTION_INITIAL_POPULATION_SIZE:-8}"
export EVOLUTION_INITIAL_POPULATION_ATTEMPTS="${EVOLUTION_INITIAL_POPULATION_ATTEMPTS:-200}"
export EVOLUTION_INITIAL_POPULATION_VARIATION="${EVOLUTION_INITIAL_POPULATION_VARIATION:-0.05}"
export EVOLUTION_INITIAL_POPULATION_LENGTH="${EVOLUTION_INITIAL_POPULATION_LENGTH:-0.02}"
export EVOLUTION_TASKS='Isaac-EvolutionHand-Grasp-v0,Isaac-EvolutionHand-BranchGrasp-v0,Isaac-EvolutionHand-Forage-v0,Isaac-EvolutionHand-Strike-v0'
export DISABLE_DEFAULT_GROUND_PLANE=1

mkdir -p "$EVOLUTION_LOG_ROOT/evolution_task" "$EVOLUTION_ROOT/parallel_eval_slots"
cd "$EVOLUTION_ROOT/Isaaclab_other"
exec "$ISAAC_ENV_PREFIX/bin/python" main_evolution.py
