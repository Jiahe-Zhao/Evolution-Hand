# Forage contact repair and Branch diagnosis

## Scope

No hand URDF, joint axes, geometry, inertia, material or morphology mutation bounds changed. Existing success distance/force/hold thresholds retained. Forage restores the planar leaf supports already declared in the mechanical scene contract; this validates the supported-leaf task, not an unconstrained free-leaf task.

## Implementation

- `forage_contact_repair.py`: one PhysX contact view per actual hand rigid body, filtering the two leaves and food separately. `ContactProxy` exposes the two leaf channels to the original rewards, termination logic, joint-graph observations and scripted controller. Whole-hand tactile features are additionally available to PPO.
- The original Forage `_setup_scene` unconditionally overwrites the contact sensor path with `Robot/.*`; changing the config before construction was insufficient. A narrowly scoped factory intercept in that task module corrects the legacy sensor at construction. The legacy scene sensor remains valid and registered, while task readings use explicit full-hand views.
- Two passive D6 joints per environment lock leaf Z and rotation relative to the kinematic food, leaving XY motion free. Contact must physically move leaves. Assert that every declared support joint exists.
- `native_goal_worker.py` makes the Forage repair and Branch whole-hand tactile path defaults for new runs (flags remain accepted), and provides `--forage_full_probe`; `native_goal_ppo.py` snapshots the repair source in new PPO runs.
- `forage_full_probe.py` runs an actual full native episode, recording pre-reset contacts, velocities, success and failure states. The Forage teacher may enter sweep after actual leaf contact, instead of waiting indefinitely for the fingertip center to reach a surface coordinate.

## Forage tests

Remote evidence root: `/home/zjh/Evolution_PC/outputs/derl_pure_ppo_20261025/contact_repair_20261027`.

1. `forage_teacher`: full 2,399-step episode failed. Contact repair worked, both leaves were touched; one leaf cleared but the script stuck in `lower`. Peak force 11.998 N; peak leaf speed 0.09925 m/s. This was a controller stage-transition problem, not proof the task was impossible.
2. `forage_teacher_contact`: changing lower-to-sweep transition to accept measured contact completed native success (1/1), episode ended at step 2,351 (~39.18 s). Final clearance distances 0.07646/0.08110 m, threshold 0.075 m; both contact flags true, invalid-motion false. Peak contact force 0.16303 N, peak leaf speed 0.008805 m/s. Last trace row is one frame after the one-shot success reward; accumulated success records that event. Future trace capture also retains all success frames explicitly.
3. Initial short PPO still emitted legacy initialization mapping errors because native setup overwrote config. That diagnostic was superseded; logs retained. `forage_short_ppo_v2` tests the corrected factory interception with 256 environments and 262,144 transitions, fresh policy, no BC. PPO hyperparameters match the previous run; this small budget is a pipeline/learning-signal check, not a convergence guarantee. Completed: untrained sample policy 254/256 (99.22%); final sample 193/256 (75.39%); final mean 197/256 (76.95%). Training itself completed 99 episodes, all successful, in 262,144 transitions (~63.2 s optimizer/rollout time, excluding full evaluations). Median first episode length increased from 686.5 to 1,143.5 steps. No convergence improvement demonstrated. Random-policy success is already near ceiling, so this scene/success metric cannot presently distinguish learned skilled clearing from exploratory pushing. Full-run diagnostics peak at 110.49 N single-body contact (maximum across leaf and food targets) and 14.59 m/s leaf speed; those include evaluations and automatically restarted, unscored episodes. A dedicated first-episode success audit is used before attributing these peaks to scored successful episodes.

## Branch — independent 256-environment full episode

`branch_batch_teacher`: 0/256 successful. No contact-filter initialization mismatch reported. The legacy sensor watches only five terminal finger links. Its sampled force is zero throughout; full-hand sensing measures real proximal contacts (overall peak 43.24 N across 256 randomized resets).

At episode end:
- Environments with force >=1 N, by thumb/index/middle/ring/little: 28/126/133/116/145.
- Negative thumb-versus-other-finger resultant dot product: 0/256.
- Maximum sampled opposition-qualified hold streak: 0; contact-only streak: 2 steps. Required: thumb + at least two long fingers, each >=1 N, for 15 consecutive steps, plus pose stability and opposing force directions.

Interpretation: old observations/reward missed proximal contact; after full-hand coverage, this teacher still does not generate opposing grasp forces. Its radial finger targets preserve the initial side of the branch and do not plan an approach around it. Contact-conditioned stopping can also stop one finger while the others never establish the required contact pattern. The kinematic branch can resist same-side pushing, so contact strength alone is not a grasp test. Peak force warrants improving the approach/controller; it is not evidence of spontaneous dynamics explosion.

Next Branch work should test branch placement relative to thumb/finger reachable surfaces and explicit opposing approach waypoints before another long PPO. Do not lower the success force/hold thresholds to turn same-side pushing into success. Current evidence does not prove the hand structure cannot grasp or that PPO cannot learn.

## Independent first-episode success quality audit

`forage_success_audit`: final sampled policy, frozen checkpoint normalization, 256 environments, original success semantics. Result 206/256 (80.47%); this fresh-process result differs from the in-training evaluator (193/256), so do not treat one seed/run as a precise generalization estimate.

- Whole audited batch: maximum single-hand-body/leaf force 240.51 N; maximum leaf speed 36.49 m/s. 18/256 first episodes contain speeds above 2 m/s; maximum 16 above-threshold frames in an episode (not necessarily consecutive).
- Successful episodes: median/p95/maximum peak single-body leaf force 26.02 / 50.43 / 240.51 N; median force at success 6.29 N. Median/p95/maximum peak leaf speed 0.460 / 2.675 / 36.49 m/s. At the success instant itself, median speed 0.0298 m/s, maximum 0.6706 m/s.
- First captured speed event: env 114, step 172, leaf speed 5.154 m/s and contact force 24.31 N, speed streak 1, invalid=false, terminated=false. This is consistent with the existing 0.15 s recovery grace: transient spikes need not terminate; later recovery can still earn success. It is not evidence that the event was absent.
- In the in-training final sampled evaluation, 62/63 failures were full-length timeouts, one ended at step 2355.

Conclusion: sensor/scene wiring is fixed and a gentle scripted success exists, but broad exploration still exposes severe force/velocity spikes. The nominal speed cap is not sufficient evidence of actual bounded post-solver motion. Thin rigid leaves, passive constraints and position-controlled contacts are candidate contributors; no A/B experiment yet isolates their individual contribution. Do not call the dynamics calibrated or restart long evolutionary training on this evidence. Next isolate support constraints vs free physical support under the same actions, then calibrate contact/control and score sustained clearing quality without rejecting every small transient. No hand structure edits were made.
