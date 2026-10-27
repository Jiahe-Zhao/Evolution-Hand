# Contact curriculum baseline — 2026-10-28

## Authorization and scope

User approved training following the Branch/Forage repairs and staged Strike proposal. Existing hand structural files remain unchanged. No BC, teacher action override, morphology search, or action lock is introduced. Preserve the successful Grasp policy as a control rather than retraining it unnecessarily.

## Task changes

- Branch: use validated whole-hand tactile mapping and opposing-contact success. Add continuous contact strength and thumb-versus-finger force opposition to observation/potential. Pure PPO controls every joint; the diagnostic teacher's proximal-contact whole-finger stop is not used or repaired here. Baseline task feasibility is not yet demonstrated.
- Forage: remove planar leaf joints in this version. A real static 0.70 m square support at z=0.100 m supports freely translating/rotating leaves under gravity. Food is a flush visual target patch, with its collision disabled to avoid duplicate coplanar support. Hand, leaf geometry/mass, and leaf contact mapping remain unchanged. Success requires at least 98% target-grid exposure, both leaves previously touched, leaf speeds below 0.15 m/s continuously for 0.5 s. Exposure uses 25×23 grid samples and conservative projected leaf bounding boxes: this can underestimate exposure for rotated leaves and is not exact pixel visibility. Brief fast movement resets the completion timer, rather than immediately failing the episode.
- Strike stage 1: actual free tool on its existing pedestal. Thumb and at least two long fingers contact; native grasp streak reaches ten steps; tool's lowest cuboid point clears its initial bottom by 10 mm; linear speed below 0.15 m/s; maintain for 0.5 s. Stage-1 success is not strike success.
- Strike stage 2: inherit stage-1 policy and observation/reward normalization, initialize a fresh optimizer, retain existing native strike success. Twenty percent of parallel environments retain the stage-1 grasp goal to reduce forgetting. Observations contain goal identity; both stages have identical input dimensions. Per-goal evaluation counts prevent reporting grasp completions as impacts.

## Training and promotion

Launcher: `launch_contact_curriculum.py`; output: `/home/zjh/Evolution_PC/outputs/derl_pure_ppo_20261025/contact_curriculum_20261028`.

1. Each task: 32 environments, 4,096 PPO steps, full-length before/after evaluations. This checks execution and finite rewards, not convergence.
2. Passing tasks: Branch, Forage, Strike stage 1, serialized on the single GPU. Each 256 environments, requested 10 million steps (actual 9,994,240), rollout 128, evaluation every 1,048,576 steps.
3. Strike stage 2 only if the final stage-1 checkpoint passes two stochastic action seeds at >=50% and mean actions at >=70%. Inherit precisely that evaluated final checkpoint. Otherwise record failed promotion and stop Strike progression; no false claim of successful grasp.
4. Stage 2 receives the same maximum training budget. No evolution population is launched until baseline feasibility is established.

PPO otherwise retained: independent 64×64 tanh actor/critic, learned exploration standard deviation, initial Adam LR 3e-4 with linear decay, gamma 0.99, GAE 0.95, clip 0.2, four optimization epochs, minibatch 512, entropy coefficient 0.01. Evaluation freezes normalization; training updates it. Full source snapshots and policy/normalization checkpoints are saved.

## Limitations and interpretation

This is a new curriculum experiment, not a demonstrated cure for nonconvergence. Random-policy performance, physical extrema, full-episode success, and per-stage success must be inspected before expanding the population. Forage is a rigid leaf proxy on a support, not a deformable leaf model. Strike retains the previously diagnosed contact force spikes as a monitored unresolved risk; this change does not certify its contact physics. The promotion test covers two action seeds, not broad object-pose perturbation robustness.
