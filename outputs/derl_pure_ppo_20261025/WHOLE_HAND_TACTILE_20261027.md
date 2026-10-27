# Full-hand tactile sensing and Branch validation

## Scope

Added an opt-in `--whole_hand_tactile` interface to the native goal worker. No URDF, collision geometry, joint limits, material, actuator settings, or ongoing training run was changed. The current Branch baseline has 22 runtime rigid bodies. Fixed palm geometry merged by the URDF importer is sensed through its surviving rigid body; this does not provide independent spatial pressure pixels over that body.

Each body reports target-filtered contact resultant, total contact resultant, force in the body's local frame, contact flag, contact duration, and frame-to-frame force change. These are PhysX contact measurements, not a validated real tactile-camera or friction/shear model. Total contact is retained separately from target contact; individual environment and self-collision contributions are not yet classified.

Branch gains 242 tactile features: observation dimension 442 -> 684. `observation_check.json` confirms reset and physical step both return finite 684D vectors. New inputs are incompatible with the existing 442D checkpoint without explicit adaptation. No claim is made that the old policy has already learned the added tactile inputs.

Digit groups are derived from current link naming; fixed geometry is represented by its merged parent. Length/width/angle changes preserve this interface. Topology changes, particularly a second thumb, require explicit role metadata rather than assuming numeric digit 1 is the only thumb.

## Success contracts

Old fingertip success is retained separately for auditing. The first diagnostic full-hand contract counted a thumb body and at least two long fingers with >=1 N, preserving the native pose thresholds and 15-step hold (0.25 s at 60 Hz). Palm contact alone cannot pass. Maximum per-body magnitude within each digit is used so adding more links does not automatically increase its credited force.

Three independent reset seeds (2027, 2028, 2029) met this contact-only contract continuously for 3 seconds. However, thumb-versus-long-finger force cosine was +0.974, +0.972, +0.998: almost the same direction. These are **false positives for oppositional grasp**, because the kinematic branch supports the push. The `branch_tactile_hold_20261027` raw reports preserve their original `success:true` contact-only labels and must not be used as evidence of valid grasp.

The corrected contract (`branch_tactile_success_contract.json`, version 2) additionally requires a negative dot product between thumb and long-finger resultant forces. This is a physical opposition check, not a claim of force closure, free-object load bearing, or disturbance robustness. Contact-only streak and original fingertip events remain observable.

## Initial diagnostic findings

- Existing fingertip-directed teacher: no valid task success. Full-hand peak approximately 3.10 N, concentrated in long-finger proximal links `link_2_0` through `link_5_0` and thumb middle link `link_1_1`. Old fingertip readings remain zero.
- Increasing the teacher's stopping force from 0.6 N to 1.5 N did not establish a valid grasp; fingertip motion objectives do not directly regulate the actually contacting middle/proximal links.
- Uniform flexion commands 0.4/0.7/1.0/1.4 rad did not achieve the original full-hand contact-only event. The zero-flexion action met that event, but further testing exposed same-side pushing.
- Thus the earlier inference that there was no contact anywhere on the hand was incorrect. The old contact interface omitted relevant contact bodies. Broadening sensing alone does not make the old success predicate sufficient.

## Verification

- Four tactile unit tests cover proximal contact inclusion, palm exclusion from digit counts, avoiding inflated force from extra links, selective reset/history, and opposing-versus-same-direction forces (reset is one combined case).
- Five existing goal-adapter tests remain passing: 9 total.
- Real simulator checks validate actual nonzero proximal/middle contact, finite expanded observations, and old-versus-new sensor comparison.
- An arbitrary-body Jacobian diagnostic initially failed because the existing fingertip helper accepts only registered fingertips. The standalone diagnostic now computes the body-origin Jacobian from the COM Jacobian directly; failed results are retained and are not interpreted as reachability failures.

Final corrected-contract results are recorded below after the independent verification completes.

## Current interpretation

Full-hand sensing is implemented and physically exercised, but the contact-only three-seed result is not valid oppositional grasp. The corrected actual-body force-feedback probe completed 1199 steps without success (peak contact approximately 1.69 N). Failure of these bounded scripts is not proof that the morphology cannot grasp the branch. It indicates the next investigation should target thumb/branch relative placement and a collision-aware oppositional trajectory, rather than blindly extending the old zero-contact-reward PPO run.

The branch is kinematic and the hand base is fixed in this native task. Consequently, relative-pose stability alone does not establish load-bearing or force closure. The new force-direction diagnostic is necessary to expose same-side pushing under these conditions.

## Final corrected-contract validation

`branch_tactile_validated_20261027/completed.json` confirms both probes and the 684D observation check completed.

- Actual-body force feedback, seed 2027: 1199 control steps, zero valid oppositional success.
- Zero-flexion hold, seed 2028: 1199 control steps, zero valid oppositional success; sampled contact-only streak reached 1126 steps (18.77 s), while opposition-qualified streak remained zero. Corrected gate rejects sustained same-side pushing.
- Outcome: tactile coverage and expanded PPO interface verified; Branch has **not** yet been demonstrated with valid oppositional grasp. No claim of PPO convergence or mechanical impossibility.
