#!/usr/bin/env bash

set -euo pipefail

remote_host="${1:-zjh@100.81.86.51}"
remote_root="${2:-~/Evolution_PC}"
local_root="$(cd "$(dirname "$0")/.." && pwd)"

rsync -av \
  --exclude logs \
  --exclude __pycache__ \
  --exclude "*.pyc" \
  --exclude "*.log" \
  --exclude "*.bak_*" \
  --exclude "*.manualbak_*" \
  --exclude "*_runtime_state.json" \
  --exclude "exp_*.json" \
  --exclude "*_evaluation_states" \
  "${remote_host}:${remote_root}/Isaaclab_other" \
  "${remote_host}:${remote_root}/evolution_tasks" \
  "${remote_host}:${remote_root}/parallel_eval_slots" \
  "${remote_host}:${remote_root}/项目说明.md" \
  "${local_root}/"

echo "Synced source snapshot from ${remote_host}:${remote_root}"
