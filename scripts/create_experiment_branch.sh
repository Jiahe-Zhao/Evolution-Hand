#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <machine> <task> [date]"
  echo "Example: $0 server51 grasp 2026-07-22"
  exit 1
fi

machine="$1"
task="$2"
branch_date="${3:-$(date +%F)}"
branch="exp/${machine}-${task}-${branch_date}"

git rev-parse --is-inside-work-tree >/dev/null

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "main" && "${current_branch}" != "dev" ]]; then
  echo "Current branch is '${current_branch}'."
  echo "Recommended: create experiment branches from 'main' or 'dev'."
fi

git checkout -b "${branch}"
echo "Created branch: ${branch}"
