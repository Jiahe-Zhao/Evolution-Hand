# Review: `keys_runtime.py`, `single_coordinator.py`, `fit_coordinator.py`

Read-only review. No file in this directory was modified except this one. No simulation was run; the numeric checks below use CPU MuJoCo FK (`keys_ik.mjb`, `pads.npz`) and the existing logs.

## Blocking issues

### B1. The approach ramp drives the index pad ~2 mm into key1 (likely run-killer)

`single_coordinator.py:22` ramps every joint from `0` to `base` with a cosine in **joint space**. That path is not a Cartesian lift, and the fitted `base` is only 3.06 mm above key1. At mid-ramp the index pad swings down through the key.

Signed clearance of the index pad to key1, same transform as `fit_coordinator.tips()`, 1137 pad anchor particles:

| ramp fraction | clearance |
|---|---|
| 0.60 | +9.61 mm |
| 0.68 | −0.94 mm |
| **0.70** | **−2.00 mm** |
| 0.72 | −2.00 mm |
| 0.75 | −0.92 mm |
| 0.78 | +0.35 mm |
| 0.80 | +1.23 mm |
| 0.90 | +2.43 mm |
| 1.00 (base) | +3.06 mm |
| press | −1.99 mm (intended) |

Peak penetration is at fraction ≈0.70, i.e. t≈0.25 s, before the first 0.1 s print. The log shows the aftermath: `SINGLE 0.310 approach [1.68e+03 …]`, then `SINGLE 0.410 hover [-1.17e+07 …]`, then `nan` at 0.51 s and `solver iterations limit reached`. The run never reaches `DONE`, so no npz/CSV is written.

Fix direction: start the hand at `base` (settle there) instead of q=0, or insert a pre-shape/Cartesian waypoint that keeps the index pad above z≈0.101 until the press. Do not change key geometry or stiffness.

### B2. Rigid force attribution is signed wrongly and duplicated

`single_coordinator.py:11-13` appends the **same** `force[i]` for both `(a,b)` and `(b,a)`. Newton stores the contact force **on shape0** (`_vendor/newton/_src/solvers/mujoco/kernels.py:1521-1522`; the package's own `runtime.py:rigid_tactile()` uses `[(a,b,+1),(b,a,-1)]`). Consequences:

- When the hand is shape1, the recorded value is the force on the **key**, not on the hand — opposite sign to the row where the hand is shape0.
- Each physical contact produces two CSV rows with identical magnitude, so aggregating by `key` double-counts.
- The `key` column is not a consistent "force on key".

Fix: sign by ownership (`+1` if `own==a[i]` else `−1`), emit one row per contact, or use `h.rigid_tactile()['force_on_body_N']`.

### B3. Soft force is the force on the hand, not on the key

`single_coordinator.py:14-19` writes `h.out_f` as the key force. `out_f` is the VBD force **on the soft tissue** (`contact.py` `_eval_soft_ef_contact`; `tactile()` docstring; `contact_reaction` applies `−forces[i]` to the rigid body). Magnitude is right, but for "force on key" the sign must be negated, and the column semantically means hand-side force.

### B4. Only key1 is pressable; keys 2–4 press == base

`fit_coordinator.py:17-23` solves each finger independently. In `coordinator_plan.json`, `press − base` is `[-0.052, 0.129, −0.219, −0.133, 0, 0, …, 0]`: fingers 3–5 are unchanged. Finger3 base already has `digit_3_flex_2 = digit_3_flex_3 = 1.5` (joint upper bound), and the press solve clips at `hi = 1.5 − 1e-5`, so key2 cannot be reached with this mount. The current single-key test only uses `press[0]`, so this is not the cause of the current failure, but it blocks the four-key task.

## Hand mount, qpos and control — what checks out

- Action order: `smoke.log` `ACTIONS` is the 21-name `.mjb` qpos order (`arch_2, digit_2_flex_1..3, … thumb_axial, digit_1_flex_1..2`).
- Lengths: `fit_coordinator` solves 16 finger values; `single_coordinator:5` appends 5 thumb zeros → 21. Correct.
- Sign convention: `joint_convention.json` maps all finger/thumb joints to `+1` (only `tissue` is `−1`), and `runtime.py:set_targets` applies the sign. Passing MuJoCo qpos as canonical is correct; all base/press values are inside limits.
- Rotation: `R_x(−90)` maps `+Z→+Y`, so index→key1 (y=+.001) and pinky→key4 (y=−.056). The reversed-order concern in `COORDINATOR_NOTE.md` does not apply to this rotation.
- Root transform: the `.mjb` root has a 180° rotation about Y at `[0,0,.3]`; `tips()` applies the full `xmat` and subtracts only the translation, so the fit is self-consistent.
- Control: `advance(.01)` = 40 substeps; `set_targets` mutates the array the CUDA graph reads; `key_q` indexing matches the observed ~0.187 mm rest travel.

## Minor / latent

- `keys_runtime._add_brush` never sets `brush_spec`, `brush_root_body`, `brush_joint_count`, etc., so inherited `brush_info()` would raise. Not called by `single_coordinator`.
- Deck top is z=0.082; keys start at z=0.094 — they float 12 mm above the deck (cosmetic).
- Key travel is absolute and includes the 0.187 mm gravity rest; "press ≥2 mm / release ≤0.5 mm" should be measured relative to rest.
- Soft pads exist only on palm + index (`region_names`, `pads.npz bone_names`), so fingers 3–5 can never report soft-pad forces. Asset limitation, not a code bug.
- Fit saturation: position x = −0.18 is exactly the upper bound; finger3 is at the joint bound (B4).
- `poses.append(h.state.body_q.numpy())` is safe on the default `cuda:0` device (host copy); it would alias across steps only if forced onto CPU.

## Verified vs unverified

- Verified: action order, 16+5 length, sign convention, rotation/finger order, root 180° handling, base clearance (+3.06 mm), the FK penetration profile.
- Unverified: that `add_usd(xform=mount)` composes `mount * USD_root` (the fit and `smoke.py position=(0,0,.3)` are consistent with it, but it was not traced through the USD loader). B1 is the supported cause of the blow-up, not a proven single cause.

## Suggested fix order

1. Remove the approach penetration (start at `base` or add a lifted waypoint) — without this no run completes.
2. Fix rigid force sign/duplication (B2) and soft force sign (B3) before using contact data.
3. Either regenerate presses for fingers 3–5 or explicitly scope the test to key1 (B4).
4. Re-run and verify press/release travel against the 0.187 mm rest.
