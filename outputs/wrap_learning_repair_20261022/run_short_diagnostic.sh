#!/usr/bin/env bash
# Reproduce the final objective-aligned diagnostic; success is not guaranteed.
# Does not resume main_evolution.py.
set -euo pipefail
repair_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repair_python=/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python
cd "$repair_dir"
if [ "$#" -ne 2 ]; then
  echo 'Usage: run_short_diagnostic.sh BC_CHECKPOINT NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$repair_python" repair_learning.py \
  --headless --plans ../contact_control_calibration_20261021/plans_wrap_cover_selected.json \
  --pad-face ymixed --kp 3 --kd .3 --contact-k 1500 --hand-gravity \
  --num-envs 256 --epochs 256 --critic-warmup 128 --critic-lr .0003 \
  --episode-steps 1500 --eval-steps 1500 --completion-reward --gamma .9995 --push \
  --residual-bc --bc-prior "$1" --output "$2" --mode ppo \
  --finger-std .0005 --wrist-std .0001 --stable-kl --ppo-lr .0000001 --seed 42
