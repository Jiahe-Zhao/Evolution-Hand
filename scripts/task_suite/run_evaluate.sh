#!/usr/bin/env bash
set -euo pipefail
task="$1"
checkpoint="$2"
shift 2
root="${EVOLUTION_CODE_ROOT:-/home/zjh/Evolution_PC}"
isaac="${ISAACLAB_ROOT:-/home/zjh/IsaacLab}"
output_root="${EVOLUTION_EVAL_ROOT:-${root}/outputs/task_suite/evaluation}"
source "${EVOLUTION_CONDA_BASE:-/home/zjh/miniconda3}/etc/profile.d/conda.sh"
conda activate "${EVOLUTION_CONDA_ENV:-evolution_isaaclab}"
exec "${isaac}/isaaclab.sh" -p "${root}/evolution_tasks/task_suite/evaluate_task.py" --headless --task "${task}" --checkpoint "${checkpoint}" --output_dir "${output_root}/${task}" "$@"
