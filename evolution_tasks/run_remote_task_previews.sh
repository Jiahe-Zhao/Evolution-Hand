#!/usr/bin/env bash
set -euo pipefail

export TERM=xterm
source /home/zjh/miniconda3/etc/profile.d/conda.sh
conda activate evolution_isaaclab
cd /home/zjh/IsaacLab

for task in branch grasp manipulation strike stone; do
  case "$task" in
    branch)
      out="/home/zjh/Evolution_PC/evolution_tasks/task_branch_grasp/scene_preview.png"
      ;;
    grasp)
      out="/home/zjh/Evolution_PC/evolution_tasks/task_grasp/scene_preview.png"
      ;;
    manipulation)
      out="/home/zjh/Evolution_PC/evolution_tasks/task_manipulation/scene_preview.png"
      ;;
    strike)
      out="/home/zjh/Evolution_PC/evolution_tasks/task_strike/scene_preview.png"
      ;;
    stone)
      out="/home/zjh/Evolution_PC/evolution_tasks/task_stone/scene_preview.png"
      ;;
  esac

  printf "RUNNING %s -> %s\n" "$task" "$out"
  timeout 240 ./isaaclab.sh -p /home/zjh/Evolution_PC/evolution_tasks/capture_evolution_task_preview.py --headless --enable_cameras --task-key "$task" --output "$out"
  ls -lh "$out"
done
