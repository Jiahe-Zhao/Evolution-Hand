#!/usr/bin/env python3
'Smoke test: AX50 VisualSkin hand sharing one Newton world with the brush14 prop.\n\nThis does NOT grip or write. It only checks that the hand and the 130 passively\nsprung brush hinges step together without one control path overwriting the other.\n\nRun:\n  PYTHONPATH=<checkout>/outputs/ax50_visualskin_v3_0_0   python smoke.py\n'

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json
import sys
import time
from pathlib import Path

import numpy as np

from adapter import AX50BrushRuntime, DEFAULT_BRUSH_XML

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "nov20" / "ax50_brush"
DT = 0.00025
WRIST = "six_axis"
PAPER_TOP_Z = 0.016032433
# Staging pose only: the brush14 paper slab sits at z~0.015 and the AX50 hand
# hangs ~0.056 m below its root, so the default root at the origin puts the
# fingers through the paper. Lift the whole hand 10 cm; this is NOT a grasp pose.
HAND_POSITION = (0.0, 0.0, 0.10)


def log(msg):
    print(msg, flush=True)


def brush_body_mask(names):
    return np.array([n == "brush" or n.startswith("bristle_") for n in names], dtype=bool)


def contact_stats(runtime, other_name):
    c = runtime.rigid_observation
    n = int(c.rigid_contact_count.numpy()[0])
    if n == 0:
        return {"count": 0, "force_max_N": 0.0, "force_sum_N": 0.0}
    sb = runtime.model.shape_body.numpy()
    names = np.array(runtime.body_names_all, dtype=object)
    s0 = c.rigid_contact_shape0.numpy()[:n]
    s1 = c.rigid_contact_shape1.numpy()[:n]
    b0 = np.where(sb[s0] >= 0, names[np.maximum(sb[s0], 0)], "")
    b1 = np.where(sb[s1] >= 0, names[np.maximum(sb[s1], 0)], "")
    brush0 = np.array([x == "brush" or x.startswith("bristle_") for x in b0])
    brush1 = np.array([x == "brush" or x.startswith("bristle_") for x in b1])
    other0 = b0 == other_name
    other1 = b1 == other_name
    pair = (brush0 & other1) | (brush1 & other0)
    force = c.force.numpy()[:n, :3]
    mag = np.linalg.norm(force, axis=1)
    return {
        "count": int(pair.sum()),
        "force_max_N": float(mag[pair].max()) if pair.any() else 0.0,
        "force_sum_N": float(mag[pair].sum()),
    }


def body_state(runtime):
    bq = runtime.state.body_q.numpy()
    names = runtime.body_names_all
    root = names.index("brush")
    tips = list(runtime.brush_tip_bodies)
    radii = [runtime.brush_spec["bundles"][b][-1]["radius"] for b in range(len(runtime.brush_spec["bundles"]))]
    tip_low = min(bq[i, 2] - 2.0 * runtime.brush_spec["bundles"][j][-1]["half_height"] - radii[j] for j, i in enumerate(tips))
    return {
        "brush_root_pos": bq[root, :3].tolist(),
        "brush_root_z": float(bq[root, 2]),
        "tip_lowest_z": float(tip_low),
        "hand_min_z": float(bq[: runtime.hand_body_count, 2].min()),
    }


def core_checks(runtime, label, failures):
    q = runtime.state.joint_q.numpy()
    qd = runtime.state.joint_qd.numpy()
    jf = runtime.control.joint_f.numpy()
    finite = bool(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(jf).all())
    hand_q = float(np.abs(q[: runtime.hand_dof_count]).max())
    hinge_q = float(np.abs(q[runtime.brush_hinge_q]).max())
    hand_f = float(np.abs(jf[: runtime.hand_dof_count]).max())
    brush_f = float(np.abs(jf[runtime.hand_dof_count:]).max())
    if not finite:
        failures.append(f"{label}: non-finite state")
    if brush_f != 0.0:
        failures.append(f"{label}: brush joint_f {brush_f} != 0 (PD leaked into brush)")
    return {
        "finite": finite,
        "max_abs_q": float(np.abs(q).max()),
        "max_abs_qd": float(np.abs(qd).max()),
        "hand_joint_max_abs_q": hand_q,
        "brush_hinge_max_abs_q": hinge_q,
        "hand_joint_f_max": hand_f,
        "brush_joint_f_max": brush_f,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    failures = []
    result = {
        "package_version": None,
        "side": "right",
        "wrist": WRIST,
        "phase_c_wrist": "locked",
        "brush_xml": str(DEFAULT_BRUSH_XML),
        "hand_position": list(HAND_POSITION),
        "dt": DT,
    }

    log("[build] zero-gravity instance")
    r = AX50BrushRuntime(
        side="right", wrist=WRIST, gravity=(0.0, 0.0, 0.0), position=HAND_POSITION
    )
    info = r.brush_info()
    result.update(info)
    log("[build] " + json.dumps(info, sort_keys=True))
    stiff = r.model.mujoco.dof_passive_stiffness.numpy()[r.brush_hinge_dof]
    result["hinge_passive_stiffness_min"] = float(stiff.min())
    result["hinge_passive_stiffness_max"] = float(stiff.max())
    result["joint_target_mode_abs_sum"] = float(np.abs(r.model.joint_target_mode.numpy()).sum())
    if not np.allclose(stiff, 4.545e-4):
        failures.append("hinge passive stiffness is not 4.545e-4 N m/rad")
    if result["joint_target_mode_abs_sum"] != 0.0:
        failures.append("joint_target_mode not all NONE")

    init = body_state(r)
    result["paper_top_z"] = PAPER_TOP_Z
    result["initial_brush_root_z"] = init["brush_root_z"]
    result["initial_tip_lowest_z"] = init["tip_lowest_z"]
    result["initial_hand_min_z"] = init["hand_min_z"]
    result["initial_hand_paper_clearance_m"] = init["hand_min_z"] - PAPER_TOP_Z

    # A. zero gravity, all targets zero
    log("[A] zero gravity, 0.05 s")
    r.advance(0.05)
    a = core_checks(r, "A", failures)
    a.update(body_state(r))
    a["steps"] = 200
    a["time_s"] = 0.05
    a["brush_root_drift_m"] = float(
        np.linalg.norm(np.array(a["brush_root_pos"]) - np.array(init["brush_root_pos"]))
    )
    a["hand_rigid_tactile_max_N"] = float(
        r.rigid_tactile()["sum_force_magnitudes_N"].max()
    )
    result["phase_a"] = a
    log("[A] " + json.dumps(a, sort_keys=True))
    if a["brush_root_drift_m"] > 1e-3:
        failures.append(f"A: brush root drifted {a['brush_root_drift_m']} m with no gravity")
    if a["brush_hinge_max_abs_q"] > 1e-2:
        failures.append("A: brush hinge moved under zero gravity")
    if a["hand_joint_max_abs_q"] > 0.01:
        failures.append("A: hand joints left zero target by more than 0.01 rad")
    if a["hand_rigid_tactile_max_N"] != 0.0:
        failures.append("A: hand reported external rigid load in the staged pose")

    # B. PD isolation: drive one hand joint, brush must stay put
    log("[B] drive one hand joint, 0.01 s")
    name = next(n for n in r.action_names if n.startswith("digit_") and "_flex" in n)
    idx = r.action_names.index(name)
    before = float(r.observe()["joint_position"][idx])
    r.set_targets({name: 0.1})
    r.advance(0.01)
    b = core_checks(r, "B", failures)
    b.update(body_state(r))
    b["driven_joint"] = name
    b["driven_joint_before_rad"] = before
    b["driven_joint_after_rad"] = float(r.observe()["joint_position"][idx])
    b["driven_joint_delta_rad"] = b["driven_joint_after_rad"] - before
    b["steps"] = 40
    b["time_s"] = 0.01
    b["brush_root_drift_m"] = float(
        np.linalg.norm(np.array(b["brush_root_pos"]) - np.array(init["brush_root_pos"]))
    )
    result["phase_b"] = b
    log("[B] " + json.dumps(b, sort_keys=True))
    if b["driven_joint_delta_rad"] <= 0.005:
        failures.append("B: hand joint did not follow its target")
    if b["brush_hinge_max_abs_q"] > 1e-2:
        failures.append("B: brush hinge responded to the hand PD command")

    # D. reset returns the whole shared world to the initial state
    log("[D] reset + 0.0025 s")
    r.reset()
    d0 = body_state(r)
    r.advance(0.0025)
    d = core_checks(r, "D", failures)
    d.update(body_state(r))
    d["steps"] = 10
    d["time_s"] = 0.0025
    d["reset_root_error_m"] = float(
        np.linalg.norm(np.array(d0["brush_root_pos"]) - np.array(init["brush_root_pos"]))
    )
    d["reset_hinge_max_abs_q"] = float(np.abs(r.state.joint_q.numpy()[r.brush_hinge_q]).max())
    result["phase_d"] = d
    log("[D] " + json.dumps(d, sort_keys=True))
    if d["reset_root_error_m"] > 1e-5:
        failures.append("D: reset did not restore the brush root pose")

    # C. gravity: brush falls onto the paper; the hand must stay untouched.
    # The six-axis wrist translations are force-capped at 1 N, which cannot hold
    # the hand's own weight, so this instance locks the wrist to isolate the
    # brush-paper contact from that sag. A/B/D still exercise the 27 actions.
    log("[C] gravity -9.81, locked wrist, new instance, 0.3 s")
    t_c = time.perf_counter()
    rg = AX50BrushRuntime(
        side="right", wrist="locked", gravity=(0.0, 0.0, -9.81), position=HAND_POSITION
    )
    c0 = body_state(rg)
    samples = []
    first_contact_s = None
    for _ in range(120):
        rg.advance(0.0025)
        cs = contact_stats(rg, "paper")
        bs = body_state(rg)
        hs = float(rg.rigid_tactile()["sum_force_magnitudes_N"].max())
        t = round((len(samples) + 1) * 0.0025, 4)
        samples.append(
            {
                "t_s": t,
                "root_z": bs["brush_root_z"],
                "tip_low_z": bs["tip_lowest_z"],
                "paper_contacts": cs["count"],
                "paper_force_max_N": cs["force_max_N"],
                "hand_rigid_tactile_max_N": hs,
            }
        )
        if cs["count"] and first_contact_s is None:
            first_contact_s = t
    c = core_checks(rg, "C", failures)
    c.update(body_state(rg))
    c["steps"] = 1200
    c["time_s"] = 0.3
    c["wrist"] = "locked"
    c["initial_brush_root_z"] = c0["brush_root_z"]
    c["initial_tip_lowest_z"] = c0["tip_lowest_z"]
    c["brush_root_drop_m"] = c0["brush_root_z"] - c["brush_root_z"]
    c["first_paper_contact_s"] = first_contact_s
    c["paper_contact_max_count"] = max(s["paper_contacts"] for s in samples)
    c["paper_contact_max_force_N"] = max(s["paper_force_max_N"] for s in samples)
    c["hand_rigid_tactile_max_N"] = max(s["hand_rigid_tactile_max_N"] for s in samples)
    c["paper_contact"] = contact_stats(rg, "paper")
    c["table_contact"] = contact_stats(rg, "table")
    c["samples"] = samples[::10] + [samples[-1]]
    c["wall_time_s"] = time.perf_counter() - t_c
    c["ms_per_step"] = 1000.0 * c["wall_time_s"] / c["steps"]
    result["phase_c"] = c
    log("[C] " + json.dumps({k: v for k, v in c.items() if k != "samples"}, sort_keys=True))
    if c["brush_root_drop_m"] <= 0.005:
        failures.append("C: brush did not fall under gravity")
    if c["paper_contact_max_count"] == 0:
        failures.append("C: no brush-paper contact during the fall")
    if c["hand_rigid_tactile_max_N"] != 0.0:
        failures.append("C: hand reported external rigid load (brush touched the hand)")
    if c["hand_joint_max_abs_q"] > 0.01:
        failures.append("C: hand joints moved under gravity with zero targets")

    result["wall_time_s"] = time.perf_counter() - t_start
    result["failures"] = failures
    result["status"] = "pass" if not failures else "fail"
    (OUT_DIR / "smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log("[result] status=%s failures=%s" % (result["status"], failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
