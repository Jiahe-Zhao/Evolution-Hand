# Branch opposition verification — 2026-10-28

## Method

No hand structure, joint limits, scene placement, collisions, controller bounds, or success thresholds changed. A diagnostic forward-kinematics model was constructed from the exact URDF and calibrated against simulator fingertip positions. Maximum error was 5.33e-7 m (0.000533 mm). Fixed finger-pad links merged into parent bodies during import were resolved through their original URDF frames, rather than assuming one runtime body per pad.

Two searches each used 512 initial joint configurations, 400 Adam updates, and actual controller joint bounds:

1. Strict: 16 radial approach directions; thumb and long fingers target opposite sides of the cylinder.
2. Relaxed: optimize direction continuously; long fingers can deviate up to 80 degrees from exact opposition. Allow pad-center radial offsets within the pad's circumscribed radius, a deliberately optimistic geometric allowance. These targets are necessary geometric proxies, not collision-free grasp certificates.

Objective targets the distal finger pads: thumb and the best two long fingers, with a small regularizer for the other fingers. It does not exhaust all possible proximal/middle-link grasp configurations. Search is nonconvex and finite; failure is not a proof of mathematical impossibility.

The 32 best candidate joint targets from each search were replayed in the unchanged physical scene, through ordinary bounded joint actions. Each was approached over either 360 or 900 control steps (6 or 15 seconds), within the native approximately 20-second episode. No joint teleport, disabled collision, or object attachment was used. These are 32 candidate postures, not 32 independent morphology/object test cases.

## Results

| Search | Geometric candidates within 3 mm | Physical simultaneous thumb + two-long contact | Opposed contact | Task success |
|---|---:|---:|---:|---:|
| Strict, 6 s ramp | 0/512 | 32/32 | 0/32 | 0/32 |
| Strict, 15 s ramp | same search | 32/32 | 0/32 | 0/32 |
| Relaxed, 6 s ramp | 0/512 | 32/32 | 0/32 | 0/32 |
| Relaxed, 15 s ramp | same search | 32/32 | 0/32 | 0/32 |

Best strict radial target errors (thumb/index/middle/ring/little): approximately 0 / 18.46 / 18.88 / 24.30 / 12.77 mm.

Best relaxed errors: approximately 0 / 9.16 / 9.39 / 15.74 / 5.92 mm. Even allowing oblique contact and optimistic pad dimensions, the search did not place two long-finger pads opposite the thumb within the 3 mm criterion.

Physical replay peak digit force was approximately 31.4–33.2 N. Opposed hold duration remained zero at both approach speeds. Thus a slower direct approach by itself does not fix the observed same-side pressing. Higher force or reduced hold duration would not constitute verification of opposition.

## Conclusion and next step

Distal-pad opposed grasp has **not been demonstrated for the present hand placement and branch placement**. The evidence points toward a workspace/placement mismatch that should be tested before more PPO, but does not establish an intrinsically incapable hand or prove that every alternate grasp is impossible.

Recommended next comparison: retain the current scene as control; search small branch translations in the unchanged hand's common thumb/long-finger workspace, verify collision-free approach and opposed hold physically, then choose an initial-pose distribution for training. Keep native opposed-contact success. Do not modify the hand structure or weaken success to same-side contact on the basis of this result.

## Evidence

Remote strict run: `outputs/derl_pure_ppo_20261025/branch_reachability_v2_20261028`.
Remote relaxed run: `outputs/derl_pure_ppo_20261025/branch_reachability_relaxed_20261028`.
Initial diagnostic failed on runtime-body versus fixed-pad lookup; that failure was retained and the mapping corrected before the above runs.
Local evidence: `branch_reachability_evidence_20261028/`.
