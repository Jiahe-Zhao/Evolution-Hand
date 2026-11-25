"""Short Newton GPU smoke for the AX50 + leaf-04 derived model.

The hand is mounted far from the leaves on purpose: this validates the derived
runtime (object import, leaf DOFs, passive torque, contact buffers) only. It is
not a hand-contact or task-success result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from leaf_runtime import LeafRuntime, OUT_DIR  # noqa: E402

HAND_POSITION = (0.25, 0.25, 0.35)
DT = 0.00025


def _finite(*arrays):
    bad = []
    for name, arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if not np.isfinite(a).all():
            bad.append(name)
    return bad


def _contact_snapshot(h):
    soft = int(h.contacts.soft_contact_count.numpy()[0])
    raw = int(h.raw_contacts.soft_contact_count.numpy()[0])
    rigid = int(h.rigid_observation.rigid_contact_count.numpy()[0])
    counts = getattr(h.vbd, "body_particle_contact_counts", None)
    vbd_per_body = int(counts.numpy().max()) if counts is not None else 0
    return soft, raw, rigid, vbd_per_body


def run_smoke(h, steps, dt, sample_every=10):
    q = h.state.joint_q.numpy()
    qd = h.state.joint_qd.numpy()
    body_q = h.state.body_q.numpy()
    mass = h.model.body_mass.numpy()

    leaf_q0 = q[h.leaf_q_np].copy()
    leaf_bodies = np.array(h.leaf_bodies, dtype=int)
    hand_bodies = np.arange(h.hand_body_count, dtype=int)
    hand_xyz = body_q[hand_bodies, :3]
    leaf_xyz = body_q[leaf_bodies, :3]
    min_hand_leaf = float(
        np.linalg.norm(hand_xyz[:, None, :] - leaf_xyz[None, :, :], axis=2).min()
    )

    peaks = {"soft": 0, "raw": 0, "rigid": 0, "vbd_per_body": 0}
    samples = []
    t_wall = time.time()
    for i in range(steps):
        h.step(dt)
        soft, raw, rigid, vbd_per_body = _contact_snapshot(h)
        peaks["soft"] = max(peaks["soft"], soft)
        peaks["raw"] = max(peaks["raw"], raw)
        peaks["rigid"] = max(peaks["rigid"], rigid)
        peaks["vbd_per_body"] = max(peaks["vbd_per_body"], vbd_per_body)
        if i % sample_every == 0 or i == steps - 1:
            qs = h.state.joint_q.numpy()
            samples.append(
                {
                    "step": i + 1,
                    "t": round(h.time, 6),
                    "leaf_q_rad": [float(qs[j]) for j in h.leaf_q_np],
                    "leaf_q_deg": [
                        float(np.degrees(qs[j])) for j in h.leaf_q_np
                    ],
                    "soft_contacts": soft,
                    "rigid_contacts": rigid,
                    "vbd_per_body_contacts": vbd_per_body,
                }
            )
        if (i + 1) % 40 == 0:
            print(
                "[smoke] step %d/%d t=%.4f leaf_q=%s soft=%d rigid=%d"
                % (
                    i + 1,
                    steps,
                    h.time,
                    np.round(h.state.joint_q.numpy()[h.leaf_q_np], 4).tolist(),
                    soft,
                    rigid,
                ),
                flush=True,
            )
    wall = time.time() - t_wall

    q1 = h.state.joint_q.numpy()
    qd1 = h.state.joint_qd.numpy()
    body_q1 = h.state.body_q.numpy()
    body_qd1 = h.state.body_qd.numpy()
    pq1 = h.state.particle_q.numpy()
    pqd1 = h.state.particle_qd.numpy()
    bad = _finite(
        ("joint_q", q1),
        ("joint_qd", qd1),
        ("body_q", body_q1),
        ("body_qd", body_qd1),
        ("particle_q", pq1),
        ("particle_qd", pqd1),
        ("body_mass", mass),
    )

    overflow = {
        "soft_contact": peaks["soft"] > int(h.contacts.soft_contact_max),
        "raw_soft_contact": peaks["raw"] > int(h.raw_contacts.soft_contact_max),
        "rigid_contact": peaks["rigid"] > int(h.rigid_observation.rigid_contact_max),
        "vbd_body_particle_per_body": peaks["vbd_per_body"] > 32768,
    }

    return {
        "steps": steps,
        "dt_s": dt,
        "sim_time_s": float(steps * dt),
        "wall_time_s": wall,
        "wall_time_per_step_s": wall / steps,
        "gpu": wp.get_device().name,
        "finite": len(bad) == 0,
        "nonfinite_fields": bad,
        "leaf_q_start_rad": [float(x) for x in leaf_q0],
        "leaf_q_end_rad": [float(q1[j]) for j in h.leaf_q_np],
        "leaf_q_end_deg": [float(np.degrees(q1[j])) for j in h.leaf_q_np],
        "leaf_qd_end": [float(qd1[j]) for j in h.leaf_dof_np],
        "min_hand_leaf_body_distance_m": min_hand_leaf,
        "contact_peak": peaks,
        "contact_limits": {
            "soft_contact_max": int(h.contacts.soft_contact_max),
            "raw_soft_contact_max": int(h.raw_contacts.soft_contact_max),
            "rigid_contact_max": int(h.rigid_observation.rigid_contact_max),
            "vbd_rigid_body_particle_contact_buffer_size": 32768,
        },
        "contact_overflow": overflow,
        "contact_overflow_any": any(overflow.values()),
        "samples": samples,
    }


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def run_probe(h, dt=1.0 / 480.0, duration=1.3):
    """Model probe: +-0.0038 Nm pulse on one leaf at a time, hand far away."""
    steps = int(round(duration / dt))
    h.extra_leaf_force = wp.zeros(h.leaf_dof_count, dtype=float)
    results = []
    for k in range(h.leaf_dof_count):
        h.extra_leaf_force.zero_()
        q0 = h.state.joint_q.numpy()[h.leaf_q_np].copy()
        peak = float(q0[k])
        series = []
        for i in range(steps):
            t = (i + 1) * dt
            amp = 0.0038 * _smooth((t - 0.15) / 0.4) * (1.0 - _smooth((t - 1.1) / 0.2))
            extra = np.zeros(h.leaf_dof_count, dtype=np.float32)
            extra[k] = amp
            h.extra_leaf_force.assign(extra)
            h.step(dt)
            q = h.state.joint_q.numpy()[h.leaf_q_np]
            peak = max(peak, float(q[k]))
            if i % 48 == 0 or i == steps - 1:
                series.append(
                    {"t": round(h.time, 5), "q_deg": float(np.degrees(q[k]))}
                )
        results.append(
            {
                "leaf": h.leaf_joint_names[k],
                "q_start_rad": float(q0[k]),
                "q_peak_rad": peak,
                "q_peak_deg": float(np.degrees(peak)),
                "series": series,
            }
        )
        print(
            "[probe] %s peak %.4f rad" % (h.leaf_joint_names[k], peak), flush=True
        )
    h.extra_leaf_force = None
    return {
        "kind": "model_probe_not_hand_pass",
        "note": "isolated 0.0038 Nm leaf torque, hand parked far away; no hand contact",
        "dt_s": dt,
        "duration_s": duration,
        "leaves": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = LeafRuntime(
        side="left",
        wrist="six_axis",
        position=HAND_POSITION,
        gravity=(0, 0, -9.81),
    )
    info = h.leaf_info()
    print("[smoke] build_time_s %.2f" % (time.time() - t0))
    print("[smoke] leaf_info:", json.dumps(info, indent=2, default=str))

    smoke = run_smoke(h, args.steps, args.dt)
    report = {
        "kind": "model_smoke_not_hand_pass",
        "hand_position": list(HAND_POSITION),
        "leaf_info": info,
        "smoke": smoke,
    }
    if args.probe:
        report["probe"] = run_probe(h)
        report["kind"] = "model_smoke_plus_probe_not_hand_pass"

    out = OUT_DIR / "smoke.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("[smoke] wrote", out)
    print(
        "[smoke] finite=%s overflow=%s sim=%.3fs leaf_q_deg=%s"
        % (
            smoke["finite"],
            smoke["contact_overflow_any"],
            smoke["sim_time_s"],
            np.round(smoke["leaf_q_end_deg"], 3).tolist(),
        )
    )


if __name__ == "__main__":
    main()
