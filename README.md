# Evolution Hand

Evolution Hand is the baseline repository for hand morphology evolution, multi-task evaluation, and Isaac Lab task integration. This repo keeps the reusable source code, robot descriptions, and slot overrides that should be shared across different compute nodes.

## Repository layout

- `Isaaclab_other/`: evolution entrypoints, hand generators, evaluation helpers, and Isaac Lab utility scripts.
- `evolution_tasks/`: task definitions for grasp, strike, stone, manipulation, and cube environments.
- `parallel_eval_slots/`: per-slot overrides for parallel evaluation workers.
- `scripts/`: helper scripts for GitHub SSH setup, branch creation, and remote source sync.
- `项目说明.md`: original Chinese project notes.

## What is tracked

This repository is intentionally code-first:

- tracked: Python source, shell scripts, URDF/STL/USD assets, YAML configs, and project docs
- ignored: logs, runtime state, evaluation caches, lineage snapshots, checkpoints, and temporary backups

If a future experiment needs a tracked config, store it under a dedicated path such as `configs/` instead of naming it `exp_*.json`.

## Recommended branch strategy

- `main`: stable shared baseline for all machines
- `dev`: integration branch before merging into `main`
- `exp/<machine>-<task>-<date>`: one branch per compute node or experiment line
- `fix/<topic>`: urgent fixes shared back to `main`

Examples:

- `exp/4090-grasp-2026-07-22`
- `exp/server51-stone-2026-07-22`
- `fix/runtime-resume`

## Suggested workflow

```bash
git checkout main
git pull origin main
bash scripts/create_experiment_branch.sh server51 grasp 2026-07-22
git add .
git commit -m "feat: adjust grasp evolution pipeline"
git push -u origin exp/server51-grasp-2026-07-22
```

When an experiment proves useful, merge the minimal reusable code back into `dev` or `main`. Keep raw logs and result files outside Git.

For first-time GitHub publishing after the remote repo exists:

```bash
bash scripts/push_main_to_github.sh Jiahe-Zhao Evolution-Hand
```

## Notes

- The current imported baseline came from `/home/zjh/Evolution_PC` on `100.81.86.51`.
- The original remote directory contained large experiment logs under `evolution_tasks/logs`, which are intentionally excluded from version control.
- For multi-node collaboration details, see `docs/WORKER_WORKFLOW.md`.
