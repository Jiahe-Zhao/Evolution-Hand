#!/bin/bash
set -euo pipefail
export EVOLUTION_ROOT=/home/zjh/Evolution_PC
export ISAACLAB_ROOT=/home/zjh/IsaacLab
export CONDA_PREFIX=/home/zjh/miniconda3/envs/evolution_isaaclab
export PATH="$CONDA_PREFIX/bin:$PATH"
export ACCEPT_EULA=Y PRIVACY_CONSENT=Y ISAAC_SIM_SETUP=/dev/null
export EVOLUTION_STRIKE_SCENE_PRESET=open_grasp_v4
export EVOLUTION_STRIKE_V4_TOOL_WIDTH_M=0.018 EVOLUTION_STRIKE_V4_PREGRASP=0.0
cd "$EVOLUTION_ROOT/Isaaclab_other"
exec "$CONDA_PREFIX/bin/python" evaluate_mechanical_task.py "$@"
