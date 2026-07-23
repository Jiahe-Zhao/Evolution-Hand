#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVOLUTION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export TERM=xterm

source /home/zjh/miniconda3/etc/profile.d/conda.sh
conda activate evolution_isaaclab
cd /home/zjh/IsaacLab

for task in branch grasp manipulation strike stone; do
  case "$task" in
    branch)
      out="$EVOLUTION_ROOT/evolution_tasks/task_branch_grasp/scene_preview.png"
      ;;
    grasp)
      out="$EVOLUTION_ROOT/evolution_tasks/task_grasp/scene_preview.png"
      ;;
    manipulation)
      out="$EVOLUTION_ROOT/evolution_tasks/task_manipulation/scene_preview.png"
      ;;
    strike)
      out="$EVOLUTION_ROOT/evolution_tasks/task_strike/scene_preview.png"
      ;;
    stone)
      out="$EVOLUTION_ROOT/evolution_tasks/task_stone/scene_preview.png"
      ;;
  esac

  tmp_out="${out%.png}.new.png"
  rm -f "$tmp_out"
  printf "RUNNING %s -> %s\n" "$task" "$out"
  timeout 240 ./isaaclab.sh -p "$EVOLUTION_ROOT/evolution_tasks/capture_evolution_task_preview.py" \
    --headless --enable_cameras --task-key "$task" --output "$tmp_out"
  test -s "$tmp_out"
  mv "$tmp_out" "$out"
  ls -lh "$out"
done
