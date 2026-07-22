# Version Control Notes

## Goal

Use GitHub to manage shared baseline code, while keeping node-specific experiment outputs outside the repository.

## Branch rules

- `main`: only stable code that any machine can pull and run
- `dev`: merged but not yet fully frozen changes
- `exp/<machine>-<task>-<date>`: experimental branches for one machine or one attempt
- `fix/<topic>`: focused bug fixes

## Commit style

Use short, meaningful commits:

- `feat: add stage2 grasp resume logic`
- `fix: correct slot override import path`
- `refactor: split hand mutation helpers`
- `docs: update remote sync notes`

## What should not enter Git

- training logs
- evaluation state folders
- runtime resume snapshots
- large lineage result dumps
- checkpoints and weights
- temporary backup files

## Practical recommendation

For each compute node:

1. Pull `main`.
2. Create one `exp/...` branch for the new attempt.
3. Only commit reusable code or config changes.
4. Push the branch early so the experiment line has a remote backup.
5. After validation, cherry-pick or merge the useful code into `dev` or `main`.

## Large artifacts

If you need to keep checkpoints or result snapshots, use external storage, GitHub Releases, or a separate data bucket. Do not store them in normal Git history.
