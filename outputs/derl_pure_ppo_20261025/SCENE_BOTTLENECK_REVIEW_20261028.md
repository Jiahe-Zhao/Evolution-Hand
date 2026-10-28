# Forage scene repair and Branch/Strike bottleneck audit — 2026-10-28

## Scope

User requested fixing Forage first, then analyzing Branch/Strike. No hand URDF, collision geometry, mass, inertia, joint limits, or structural mutation was changed. No extra long training was launched during this audit. The stalled Forage process (PID 1149355) had never produced a PPO update and was stopped; all old outputs remain.

## Forage: confirmed execution repair

The 256-environment run had spent over five hours in `replicationAttachEndFn -> PhysX.replicate`, before stepping physics. The custom table was created/recloned after `clone_environments` registered replication. The repair inserts the table into the source environment before cloning, allowing a single consistent source hierarchy to be copied. This removes the observed stall in the 256-environment retest; it identifies the scene construction path as the proximate cause, not an RL convergence problem. It does not prove the underlying PhysX internal failure mechanism.

Actual validation under the corrected construction order:

- 256 environments, neutral joint holding for 240 steps: no spontaneous success/contact, leaf center heights stayed 0.102–0.106 m, peak leaf speed 0.00334 m/s.
- 256 environments, random normalized actions, all first episodes completed at 2,399 steps: no early termination, no nonfinite values, both leaves contacted in every environment, zero successes. Peak aggregate hand-to-leaf force 40.95 N, peak leaf speed 1.234 m/s. This is an execution/physics smoke test, not a successful foraging demonstration or a full contact-quality certification.
- 256 environments, 32,768 PPO environment steps, 256 optimizer minibatch updates: completed, finite, training-only throughput 5,177 steps/s. This short optimizer test contains no completed training episodes and cannot establish learning success.
- The conservative exposure metric remained zero even at each episode's maximum in the random test. The scene no longer produces the old random-action near-perfect success. However the exposure-only potential also has a dead zone while both leaves still cover the whole target; future reward design should include physically meaningful clearance progress before exposure.

Progress protection is added to the two curriculum launchers. A simulation heartbeat updates every 120 vector steps, including evaluation. Ten minutes without either simulation or optimizer progress terminates only that launcher's own worker and writes `watchdog_failure.json`. Repeated stack dumps/log timestamps do not count as progress. Diagnostics use a separate four-minute no-progress watchdog. Tested normal exit and stalled-process termination locally.

## Branch: contact improves, wrapping does not

Same saved initial/final checkpoints, frozen observation normalization, 32 full first episodes each. These smaller diagnostic batches complement the earlier final 256-environment all-zero evaluation; they are not a new generalization estimate.

| Event ever reached within an episode | Initial stochastic | Final stochastic | Final mean |
|---|---:|---:|---:|
| Thumb contact >=1 N | 12/32 | 29/32 | 32/32 |
| At least two long fingers contact >=1 N | 0/32 | 15/32 | 29/32 |
| Both conditions simultaneous | 0/32 | 1/32 | 7/32 |
| Simultaneous contacts with opposing resultant forces | 0/32 | 0/32 | 0/32 |
| Final task success | 0/32 | 0/32 | 0/32 |

Contact-only streak reached 53 steps stochastically and 31 with mean actions, exceeding the 15-step duration requirement. Opposed hold remained zero. Thus simply reducing hold duration would not repair the observed failure.

Strongest forces were on long-finger proximal bodies (`link_2_0`, `link_5_0`, etc.), up to 155.49 N in this diagnostic. Last configured fingertip bodies had zero peak force in the stochastic final test and only 0.56 N in the mean test, below the 1 N contact threshold. The policy mostly presses proximal links against the fixed branch rather than producing distal wrapping and opposed grasp. High force on a fixed branch is not evidence of a stable grasp.

Confirmed bottleneck: simultaneous opposing contact and distal wrapping. Not proven: that the current hand cannot reach an opposed configuration. A geometric/IK feasibility test is still needed before blaming morphology or changing hand structure. The historical scripted teacher's whole-finger stopping rule is not used by these pure-PPO policies and must not be cited as their direct cause.

## Strike: failure precedes lift/hold

Same stage-1 checkpoint, same inputs, frozen normalization, 32 complete first episodes per mode:

| Event | Initial stochastic | Final stochastic | Final mean | Final stochastic, wrist held fixed |
|---|---:|---:|---:|---:|
| Thumb ever contacts | 25 | 9 | 3 | 13 |
| Two long fingers ever contact together | 0 | 0 | 0 | 0 |
| Thumb + two long fingers simultaneous | 0 | 0 | 0 | 0 |
| Native sustained grasp | 0 | 0 | 0 | 0 |
| Held lift / stage-1 success | 0 | 0 | 0 | 0 |

Final stochastic/mean mean episode lengths were 371.9/362.1 steps; each had 24/32 early terminations. The initial policy averaged 533.8 steps. These observations show loss of useful grasp interaction, not improvement hidden behind the 10 mm lift or 0.5 s completion threshold. Fixing wrist actions to zero does not restore long-finger contact; wrist motion alone is not sufficient explanation. That comparison is closed-loop, so subsequent finger actions can differ after state divergence.

A diagnostic fixed-wrist episode reached 42.03 m/s. A fresh repeat with independent all-hand-to-tool and tool-to-support/target contact views did not reproduce that outlier: neutral 0.164 m/s, fixed-wrist 2.332 m/s, stochastic 3.171 m/s. All-hand views found only low thumb contacts (max about 0.207 N) in that repeat. Thus the high-speed outlier remains unresolved, and cannot yet be assigned to palm, support, or target collisions. Seed equality does not guarantee bitwise deterministic PhysX reproduction. Preserve the anomaly record; do not claim the physics is fully validated or that every tool loss is an explosive contact.

## Shared reward limitation (confirmed algebraically and in traces)

The actual reward is `r_t = gamma * Phi(s_(t+1)) - Phi(s_t) + 10 * success`, with terminal potential zero and gamma 0.99. For a complete failed episode:

`sum_t gamma^t r_t = -Phi(s_0)`.

The measured residual from this identity was below 4.3e-6 in Branch and Strike audits. All failed paths from an identical initial state therefore have the same unnormalized discounted objective, even if one makes more useful contact. Potential shaping can improve transient learning/credit assignment; it does not itself turn partial completion into a higher-return objective. Approximate value learning, GAE, and online reward normalization can still change optimization behavior. This audit does not prove that avoiding the object has a larger true raw return or isolate which of those mechanisms caused the policy to drift.

At 60 Hz, gamma 0.99 discounts a success after 10 seconds by 0.002405 and after 20 seconds by 0.000005784. The short reward horizon is an additional mismatch for slow multistage contact operations; it is not proof that changing gamma alone will solve them.

## Recommended next implementation

1. Branch: verify reachable opposed targets under unchanged geometry; train proximal support plus independent distal wrapping, with continuous geometry/proximity progress before contact and actual opposed-contact progress. Do not solve zero score by accepting same-side pushing.
2. Strike: split the existing grasp stage further into contact establishment, stable grasp on support, then unsupported lift/hold. Retain free-tool physics and separate these goal-success labels from strike success. Use achievable goal curricula and bounded milestone rewards, plus retention checks, rather than requiring every first successful discovery to already complete the whole lift sequence.
3. Shared: give bounded, meaningful partial-task rewards, review temporal discount against seconds rather than copying locomotion settings unchanged, and monitor milestone curves and first-episode physics. Run limited baseline comparisons before more 10-million-step runs or population expansion.
4. Forage: runtime is repaired, but demonstration success and a useful pre-exposure learning gradient remain unverified. Do not interpret short PPO execution as successful foraging.

## Reproducible evidence

Remote root: `/home/zjh/Evolution_PC/outputs/derl_pure_ppo_20261025/scene_bottleneck_20261028`.

- `forage/bottleneck_audit.json`: 256-environment neutral + complete random episodes.
- `forage_ppo/progress.json`, `completed.json`: finite 32,768-step optimizer check.
- `branch/bottleneck_audit.json`, `strike/bottleneck_audit.json`: first-episode stage counts and discounted-return residuals.
- `strike_collision/bottleneck_audit.json`: independent contact-view repeat and its limits.
- Local copies under `scene_bottleneck_evidence_20261028/`.
