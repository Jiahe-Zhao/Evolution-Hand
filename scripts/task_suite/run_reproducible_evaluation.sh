#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 TASK CHECKPOINT EPISODES SEED [extra evaluate_task.py arguments]" >&2
  exit 2
fi

task="$1"
checkpoint="$2"
episodes="$3"
seed="$4"
shift 4
root="${EVOLUTION_CODE_ROOT:-/home/zjh/Evolution_PC}"
output_root="${EVOLUTION_REPRO_EVAL_ROOT:-${root}/outputs/reproducible_evaluation}"
episodes_dir="${output_root}/${task}/episodes"
mkdir -p "${episodes_dir}"

for ((index = 0; index < episodes; index++)); do
  episode_seed=$((seed + index))
  episode_dir="${episodes_dir}/episode_$(printf '%03d' "${index}")"
  EVOLUTION_EVAL_ROOT="${episodes_dir}" \
    bash "${root}/scripts/task_suite/run_evaluate.sh" "${task}" "${checkpoint}" \
      --episodes 1 --episode_index "${index}" --seed "${episode_seed}" \
      --output_dir "${episode_dir}" "$@"
done

"${EVOLUTION_CONDA_BASE:-/home/zjh/miniconda3}/envs/${EVOLUTION_CONDA_ENV:-evolution_isaaclab}/bin/python" \
  "${root}/evolution_tasks/task_suite/summarize_evaluations.py" \
  --task "${task}" --episodes_dir "${episodes_dir}" --output "${output_root}/${task}/evaluation.json"
