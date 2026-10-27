# Branch / Strike failure diagnosis — 2026-10-28

No hand structure, scene dynamics, reward or training checkpoint changed. Strike uses a read-only diagnostic contact view that is NOT appended to policy observations. No new PPO run launched.

## Branch: separate old training failure from current script failure

Old 9,994,240-step run: sample and mean evaluations 0/256; old fingertip contact channels remained zero. Subsequent full-hand 256-env scripted diagnostic shows real proximal contact but 0/256 success. All saved opposition samples have nonnegative thumb-versus-other-finger resultant dot products.

The current script's controller/feedback semantics conflict:

1. Teacher targets fingertip positions near the branch cylinder surface, but contacts are detected at proximal finger segments. In example env0, initial fingertip radial distances from the branch axis are 54.43/103.75/108.61/99.65/78.03 mm; branch radius is 9 mm. End distances remain 54.15/93.53/93.87/88.02/68.86 mm.
2. After full-hand feedback is substituted, any segment force >0.6 N zeroes the entire finger's Cartesian error. Proximal touch therefore stops distal wrapping before the fingertip target is reached. The script also stops below the native success threshold of 1 N. Required success is thumb plus at least two long fingers, for 15 consecutive steps, with pose stability and opposed contact forces.
3. Script radial approach directions are taken from initial positions; there is no planned opposite-side approach/regrasp. Raising the force target alone did not solve the earlier single-env probe. Stronger pushing is not an adequate fix.

These controller issues explain the scripted feasibility failure after the sensor repair; they were NOT the controller used by the old pure PPO training. Old training is compromised by missing contact signals. Evidence does not prove the hand cannot physically grasp the branch.

Recommended: define opposing reachable contact targets for proximal/middle/distal segments; maintain proximal support while allowing distal wrap, rather than stopping the whole digit at first contact. Align controller target force with native success thresholds. Validate a complete grasp before new PPO; keep hand geometry unchanged.

## Strike: four full first-episode controls, 64 environments each

Remote evidence: `outputs/derl_pure_ppo_20261025/strike_failure_audit_20261028/failure_audit.json`.
Final checkpoint from the 9,994,240-step run; frozen checkpoint observation normalization. Same original scene, 20 s episodes, open_grasp_v4, 18×18×180 mm free tool, 80 g, real gravity, 12×12×20 mm support. The fixed-wrist trial is closed-loop: finger actions can change after state divergence, so it is not exact action replay.

| Control | Ever valid stable grasp | Later lost grasp | Strike success | Peak hand/tool digit force | Peak tool speed |
|---|---:|---:|---:|---:|---:|
| Hold initial joint positions, wrist fixed | 0/64 | 0 | 0/64 | 0 N | 0.164 m/s |
| Final sampled policy, wrist fixed | 0/64 | 0 | 0/64 | 375.65 N | 17.02 m/s |
| Final sampled policy | 2/64 | 2 | 0/64 | 106.28 N | 3.37 m/s |
| Final mean policy | 2/64 | 2 | 0/64 | 275.83 N | 2.55 m/s |

- Static hold: all 64 episodes time out with the tool standing on its support. This excludes spontaneous instability of this tested initial placement; it does not validate all future contacts.
- Sample policy: 58/64 ever contact with the thumb, only 3/64 ever reach the two-long-finger condition, and only 2/64 sustain grasp. Both later lose it. Mean policy: 44 thumb / 5 long / 4 simultaneous / 2 sustained, both lost. Primary failure is establishing and maintaining multi-finger grasp, before deliberate striking.
- Native segment-contact forces match the independent whole-hand per-digit maxima to within ~1.91e-6 N for sample evaluation. Unlike Branch's old fingertip-only setup, no comparable missing-contact mechanism was observed.
- Fixed wrist does not solve failure and still permits severe spikes. Wrist motion is not the sole cause; finger/tool/support contact and finger coordination need attention.
- Many first crossings of 2 m/s occur without any prior grasp, with near-zero current hand contact, downward tool velocity and low height. They are falling-tool events, not automatically evidence of an explosive solver impulse. The 17 m/s fixed-wrist maximum and large contact forces still require separate event-level physical diagnosis.
- Impact threshold reached in 16 fixed-wrist, 11 sampled and 22 mean episodes, but no valid final success: an uncontrolled tool can hit the target without being held. Keeping the current grasp requirement is appropriate.
- Initial env0 tool center z=0.37056 m, bottom z=0.28056 m, target surface z=0.201 m. Vertical gap is ~79.6 mm. At the configured wrist speed 6 mm/s, a purely vertical wrist translation of this gap takes ~13.3 s within a 20 s episode. This estimate assumes the held tool keeps its orientation; it is not a proof of impossibility. Sequential grasp/transport/strike coordination needs sustained commands.

## Learning implications

Current goal adapter retains potential-difference shaping and a final success bonus; the native one-off grasp milestone is not directly retained. With valid grasp extremely rare, the policy gets little experience of successful held-tool transport/impact. More steps alone cannot establish that the scene and contact behavior are reliable.

Next: isolate finger closure against the support with known opposing contact targets; record peak-force context to distinguish pinching against fixtures from ordinary falls. Then validate stable grasp and low-height transport, followed by striking, using stage progress signals without rewarding indefinite grasp-only behavior. Avoid weakening final success to count dropped-tool impacts, or introducing brittle one-frame failure gates. Hand structure changes are not required for these diagnostics.
