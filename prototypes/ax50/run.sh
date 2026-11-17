#!/usr/bin/env bash
set -eu
base=$(cd "$(dirname "$0")" && pwd)
kind=${1:-}; side=${2:-}
case "$kind:$side" in basic:left|basic:right|matched:left|matched:right|hold30:) ;; *) echo "usage: $0 {basic|matched} {left|right}, or $0 hold30" >&2; exit 2;; esac
run=$(mktemp -d "${TMPDIR:-/tmp}/ax50-XXXXXX")
cp "$base"/runtime/* "$run"/
cd "$run"
echo "Run directory: $run"
cc -O2 -shared -fPIC layer.c -o layer.so -lm
export GOAL=arc_plus.npy BACK_LIMIT=-.69 SWEEP_AXIS="$(cat ax50_axis.txt)" MOUNT_FILE=ax50_mount.txt DISTAL=0 SIDE="$side"
unset FLEX_MOTION ADD_TWO
py=${NEWTON_PYTHON:-python3}
case "$kind" in
basic) export RUN="basic_$side" MOTION=ax50.npy; exec "$py" basic_check.py pinch;;
matched) export RUN="matched_$side" MOTION=upair0.npy ADD_TWO=1 EXTRA_GOALS=upair0_g.npy; exec "$py" matched_trial.py pinch;;
hold30) export RUN=hold30 MOTION=ball_place_motion.npy BALL_MM=78,36,35 CRADLE_Q=shelf_q0.npy; exec "$py" hold30_newton.py release;;
esac
