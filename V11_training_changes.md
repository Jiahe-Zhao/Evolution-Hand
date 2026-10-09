# V11: Full Strike BC Warm Start

Date: 2026-10-09
Remote repository: /home/zjh/Evolution_PC
Experiment: exp_20261009_strikephaseBC15g_v11

## Changes From V10

- Stop V10 before starting a new independent lineage.
- Strike stage1 budget: 25 -> 100 PPO iterations.
- Strike stage2 budget: 50 -> 250 PPO iterations, resumed from stage1.
- Strike BC refit: 0 -> 10 epochs on the verified complete demonstration dataset.
- Save bc_init before PPO; disable online frozen-teacher grasp takeover.
- Keep morphology/dataset hash and action/observation contract validation.
- The verified seed morphology receives BC initialization. Mutated descendants inherit mapped task policies; they do not blindly replay fixed-shape demonstrations.
- Keep Forage 25/50, 4096 environments, one parallel slot, 15 generations, population 8, stage2 top fraction 0.25, and other V10 settings.

## Inputs And Cleanup

Preserve V9 lineage, verified_seed_bc_13_14.json, Strike BC dataset/checkpoint, source code and assets. Remove only untracked/generated paths explicitly named exp_20261008_strikephaseBC15g_v10. Do not remove shared collision caches or uncertain files.

## Verification

Check shell syntax, dataset hash and full-trajectory metadata before launch. After launch verify process environment and logs. Completion of initialization is not evidence of task success; final evaluation must use the same morphology and controller, report episode success and physical quality separately.
