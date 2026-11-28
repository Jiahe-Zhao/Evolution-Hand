# Evolution Hand

Evolution Hand is the baseline repository for hand morphology evolution, multi-task evaluation, and Isaac Lab task integration. This repo keeps the reusable source code, robot descriptions, and slot overrides that should be shared across different compute nodes.

## 从零运行场景（推荐入口）

已补齐原生 Newton 场景脚本、模型、固定动作种子和锁定依赖，不需要原实验机或 IsaacLab 安装。

```bash
python3.12 -m venv .venv
.venv/bin/python scripts/bootstrap_scenes.py --install
.venv/bin/python scripts/run_scene.py --list
.venv/bin/python scripts/run_scene.py 04 --mode demo
.venv/bin/python scripts/run_scene.py ax50-04 --side right
```

Linux / Python 3.12；AX50 软体需要 NVIDIA GPU。基础刚体示例可用 CPU。启动成功和任务得分成功分别记录。

[完整安装与验证说明](docs/PORTABLE_NEWTON_SCENES.md) · [场景清单](configs/newton_scenes.json) · [47 处改动的处理](docs/review/UNCOMMITTED_47.md)

## AX50 下载与导入 / AX50 assets

**AX50 VisualSkin v3.0.0 完整包已包含在默认分支中。** 普通 `git clone` 或 GitHub 的 Download ZIP 均包含它，不需要 Git LFS，也不依赖实验机的 `outputs/` 目录。

```bash
git pull --ff-only origin main  # 已有 main checkout；有本地修改时建议另行 clone
python3 scripts/unpack_ax50.py
```

解压位置：`local_assets/ax50_visualskin_v3_0_0/`，内含左右手、Newton 运行入口、wheel、示例、验收记录和同步视频。解压器会校验完整包及其中 967 个文件。

- [完整资产包](assets/releases/AX50_VisualSkin_v3.0.0.zip)
- [导入说明及版本边界](docs/AX50_ASSET_IMPORT.md)
- [仓库同步审计：哪些已上传，哪些尚未整合](docs/GITHUB_ASSET_AUDIT_20261128.md)

本次已合入场景进展分支，并补齐统一入口所需资产。旧 AX50 原型仍与 VisualSkin 分开发行；本次不替换现有训练基准。完整实验历史、旧训练 checkpoint 不在便携运行包中。

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
