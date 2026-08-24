#!/usr/bin/env bash
set -euo pipefail
task="$1"
shift
root="${EVOLUTION_CODE_ROOT:-/home/zjh/Evolution_PC}"
isaac="${ISAACLAB_ROOT:-/home/zjh/IsaacLab}"
source "${EVOLUTION_CONDA_BASE:-/home/zjh/miniconda3}/etc/profile.d/conda.sh"
conda activate "${EVOLUTION_CONDA_ENV:-evolution_isaaclab}"
exec "${isaac}/isaaclab.sh" -p "${root}/evolution_tasks/train_interface.py" --headless --task "${task}" "$@"
