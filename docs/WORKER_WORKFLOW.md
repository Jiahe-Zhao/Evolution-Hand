# Worker Workflow

## Goal

Keep all compute nodes aligned to one shared baseline, while letting each machine run independent experiments safely.

## Initial setup on a new compute node

```bash
git clone git@github.com:<owner>/Evolution-Hand.git
cd Evolution-Hand
git checkout main
```

If the node needs its own GitHub SSH key:

```bash
bash scripts/setup_github_ssh.sh <owner> Evolution-Hand ~/.ssh/evolution_hand_github
```

For the first push from the baseline machine:

```bash
bash scripts/push_main_to_github.sh <owner> Evolution-Hand ~/.ssh/evolution_hand_github
```

## Start a new experiment

```bash
git checkout main
git pull origin main
bash scripts/create_experiment_branch.sh server51 grasp 2026-07-22
```

Recommended branch naming:

- `exp/server51-grasp-2026-07-22`
- `exp/4090-stone-2026-07-22`
- `exp/a800-multitask-2026-07-22`

## During the experiment

- commit only reusable code, task config, URDF, or evaluation logic
- do not commit logs, runtime states, checkpoints, or lineage dumps
- push the branch early so the attempt is backed up remotely

## Merge back

When one node produces a useful change:

```bash
git checkout main
git pull origin main
git merge --no-ff exp/server51-grasp-2026-07-22
git push origin main
```

If the branch contains both reusable code and local-only hacks, cherry-pick the useful commits instead of merging the whole branch.
