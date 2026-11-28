"""Run AX50 fruit-02 (pick positive & tap negative) with side-pinch opposed thumb/index."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import csv
import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from fruit_runtime import *

mode = sys.argv[1] if len(sys.argv) > 1 else "pick"
assert mode in ("pick", "tap"), f"Unknown mode: {mode}"
tag = os.environ.get("TAG", mode)
pinch_scale = float(os.environ.get("PINCH_SCALE", "1.0"))
z_off = float(os.environ.get("Z_OFF", "0.0"))

# Side-pinch mount: local +X -> world -X, local +Y -> world +Y, local +Z -> world -Z
# Index distal is ABOVE (+Z), thumb distal is BELOW (-Z), aligned in Y!
# Index presses down on fruit top during pull_down; thumb supports fruit bottom after break!
R_MOUNT = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=float)
QUAT = R.from_matrix(R_MOUNT).as_quat().tolist()

# Calibrated from fit_side_pinch.py at pos=[0.05, 0, 0.08]:
#   open mid ~ [-0.015, +0.052, +0.080] => rel offset from mount is [-0.065, +0.052, 0.000]
# Align mid to fruit equilibrium [-0.093, 0.003, 0.0808] after -35mm X approach:
FRUIT_EQ = np.array([-0.093, 0.003, 0.0808], dtype=float)
MOUNT_POS = (FRUIT_EQ - np.array([-0.068, 0.052, 0.000]) + np.array([0.035, 0.0, z_off])).tolist()

h = FruitRuntime(
    side="left",
    wrist="six_axis",
    position=MOUNT_POS,
    orientation=QUAT,
    gravity=(0, 0, -9.81),
    iterations=50,
)
h.vbd.prop_contact_ke = 450.0
h.vbd.prop_contact_kd = 0.60

i_palm = h.body_names.index("link_palm")
n_dof = len(h.action_names)

# Open pose: index above (+Z ~ +24mm), thumb below (-Z ~ -24mm), cleanly straddling 28mm fruit
pose_open = np.zeros(n_dof, dtype=float)
for b in (10, 14, 18):
    pose_open[b + 1] = 0.45
    pose_open[b + 2] = 0.45
    pose_open[b + 3] = 0.25
pose_open[7] = 0.45; pose_open[8] = 0.45; pose_open[9] = 0.25
pose_open[22] = 0.85; pose_open[23] = 0.70; pose_open[24] = 0.45; pose_open[25] = 0.65; pose_open[26] = 0.50

# Closed opposed pinch: index flexes down (-Z), thumb sweeps up (+Z) to clamp fruit tightly
pose_close = pose_open.copy()
pose_close[7] = 0.98; pose_close[8] = 0.95; pose_close[9] = 0.85
pose_close[22] = 0.25; pose_close[23] = 0.30; pose_close[24] = 0.20; pose_close[25] = 0.45; pose_close[26] = 0.75

def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

sb = h.model.shape_body.numpy()
total_time = 2.40 if mode == "pick" else 1.50
n_steps = int(round(total_time / 0.01))
rows = []
contacts_log = []
body_traj = []
particle_traj = []
stop_reason = "completed"
break_t = None
break_demand = 0.0

for step_i in range(n_steps):
    t = step_i * 0.01
    if t < 0.35:
        u = smooth(t / 0.35)
        target = pose_open * u
        d_world = np.zeros(3)
        phase = "preshape"
    elif t < 0.70:
        u = smooth((t - 0.35) / 0.35)
        target = pose_open.copy()
        d_world = np.array([-0.035 * u, 0.0, 0.0])
        phase = "approach"
    elif t < 1.15:
        u = smooth((t - 0.70) / 0.45)
        if mode == "tap":
            u_tap = 0.45 * smooth((t - 0.70) / 0.22) * (1.0 - smooth((t - 0.98) / 0.17))
            target = pose_open + (pose_close - pose_open) * u_tap
            d_world = np.array([-0.035, 0.0, 0.0])
            phase = "tap"
        else:
            target = pose_open + (pose_close - pose_open) * (u * pinch_scale)
            d_world = np.array([-0.035, 0.0, 0.0])
            phase = "pinch"
    elif mode == "pick" and t < 1.65:
        u = smooth((t - 1.15) / 0.50)
        s_now = pinch_scale + 0.08 * u
        target = pose_open + (pose_close - pose_open) * s_now
        d_world = np.array([-0.035 - 0.004 * u, 0.0, -0.038 * u])
        phase = "pull_down"
    else:
        target = (pose_open + (pose_close - pose_open) * (pinch_scale + 0.08)) if mode == "pick" else pose_open.copy()
        d_world = np.array([-0.039 if mode == "pick" else -0.035, 0.0, -0.038 if mode == "pick" else 0.0])
        phase = "hold" if mode == "pick" else "retreat"

    target[0:3] = R_MOUNT.T @ d_world
    target = np.clip(target, h.action_lower, h.action_upper)
    h.set_targets(target)
    h.advance(0.01, dt=0.00005 if t >= 0.70 else 0.000125)

    q = h.state.joint_q.numpy()
    pq = h.state.particle_q.numpy()
    pv = h.state.particle_qd.numpy()
    bq = h.state.body_q.numpy()
    st = h.stem_state.numpy()
    attached = bool(st[0] > 0.5)
    stem_demand = float(st[2])
    stem_peak = float(st[4])
    if (not attached) and break_t is None:
        break_t = float(h.time)
        break_demand = stem_peak

    finite = bool(all(np.isfinite(x).all() for x in (q, pq, pv, bq)))
    soft_v = float(np.nanmax(np.linalg.norm(pv, axis=1)))

    thumb_N = 0.0
    index_N = 0.0
    other_N = 0.0
    ext_contacts = 0

    c_rig = h.rigid_observation
    nr = int(c_rig.rigid_contact_count.numpy()[0])
    s0 = c_rig.rigid_contact_shape0.numpy()[:nr]
    s1 = c_rig.rigid_contact_shape1.numpy()[:nr]
    ff = c_rig.force.numpy()[:nr, :3]
    nm = c_rig.rigid_contact_normal.numpy()[:nr]
    for j in range(nr):
        b0, b1 = int(sb[s0[j]]), int(sb[s1[j]])
        if h.fruit_body not in (b0, b1):
            continue
        fn = abs(float(np.dot(ff[j], nm[j])))
        if fn <= 1e-6:
            continue
        other = b1 if b0 == h.fruit_body else b0
        if 0 <= other < h.hand_body_count:
            bname = h.body_names[other]
            if bname == "link_1_2":
                thumb_N += fn
            elif bname == "link_2_3":
                index_N += fn
            else:
                other_N += fn
            contacts_log.append([h.time, "rigid", bname, fn])
        else:
            ext_contacts += 1

    c_soft = h.contacts
    ns = int(c_soft.soft_contact_count.numpy()[0])
    sh = c_soft.soft_contact_shape.numpy()[:ns]
    ii = c_soft.soft_contact_indices.numpy()[:ns]
    bb = c_soft.soft_contact_barycentric.numpy()[:ns]
    sff = h.out_f.numpy()[:ns]
    snm = c_soft.soft_contact_normal.numpy()[:ns]
    for j in range(ns):
        if int(sb[sh[j]]) != h.fruit_body:
            continue
        fn = abs(float(np.dot(sff[j], snm[j])))
        if fn <= 1e-6:
            continue
        weights = {}
        for pid, w in zip(ii[j], bb[j]):
            if w > 0 and 0 <= pid < h.real_count:
                reg = h.region_names[h.regions[pid]] if h.regions[pid] >= 0 else "other"
                weights[reg] = weights.get(reg, 0.0) + float(w)
        for reg, w in weights.items():
            fp = fn * w
            if reg == "thumb_distal":
                thumb_N += fp
            elif reg == "index_distal":
                index_N += fp
            else:
                other_N += fp
            contacts_log.append([h.time, "soft", reg, fp])

    bp = bq[i_palm]
    fp_pos = bq[h.fruit_body, :3]
    rel = R.from_quat(bp[3:]).inv().apply(fp_pos - bp[:3]) if finite else np.zeros(3)
    rows.append([h.time, phase, stem_demand, int(attached), thumb_N, index_N, other_N, ext_contacts, *rel, *fp_pos, finite, soft_v])
    body_traj.append(bq)
    particle_traj.append(pq[:h.real_count])

    if step_i % 20 == 0 or step_i == n_steps - 1:
        print(
            f"[{tag}] t={h.time:.2f}s {phase:9s} att={int(attached)} stem={stem_demand:.3f}N(pk={stem_peak:.3f}) "
            f"th={thumb_N:.2f}N ix={index_N:.2f}N ext={ext_contacts} z={fp_pos[2]*1000:.1f}mm softv={soft_v:.3f}",
            flush=True,
        )
    if not finite or soft_v > 25.0:
        stop_reason = "nonfinite_or_runaway"
        break

np.savez_compressed(
    OUT_DIR / f"{tag}.npz",
    time=np.array([r[0] for r in rows]),
    body_q=np.array(body_traj),
    body_names=h.body_names_all,
    particles=np.array(particle_traj),
    tri=h.model.tri_indices.numpy(),
    attached=np.array([r[3] for r in rows]),
    stem_N=np.array([r[2] for r in rows]),
    thumb_N=np.array([r[4] for r in rows]),
    index_N=np.array([r[5] for r in rows]),
)
with (OUT_DIR / f"{tag}_trace.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "phase", "stem_demand_N", "attached", "thumb_tip_N", "index_tip_N", "other_N", "external_contacts", "rel_x", "rel_y", "rel_z", "x", "y", "z", "finite", "max_soft_speed_m_s"])
    w.writerows(rows)
with (OUT_DIR / f"{tag}_contacts.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time_s", "type", "body_or_region", "normal_N"])
    w.writerows(contacts_log)

times_arr = np.array([r[0] for r in rows])
rel_arr = np.array([[r[8], r[9], r[10]] for r in rows])
th_arr = np.array([r[4] for r in rows])
ix_arr = np.array([r[5] for r in rows])
ext_arr = np.array([r[7] for r in rows])
sel = times_arr >= times_arr[-1] - 0.51
slip_mm = float(np.linalg.norm(rel_arr[sel] - rel_arr[sel][0], axis=1).max() * 1000.0) if len(rel_arr[sel]) else float("nan")
summary = {
    "mode": mode,
    "tag": tag,
    "stop": stop_reason,
    "time_s": float(h.time),
    "finite": bool(all(r[14] for r in rows)),
    "attached_end": bool(rows[-1][3]),
    "break_time_s": break_t,
    "break_demand_N": break_demand,
    "peak_stem_demand_N": float(h.stem_state.numpy()[4]),
    "max_thumb_N": float(th_arr.max()),
    "max_index_N": float(ix_arr.max()),
    "hold_min_thumb_N": float(th_arr[sel].min()),
    "hold_min_index_N": float(ix_arr[sel].min()),
    "hold_max_ext": int(ext_arr[sel].max()),
    "hold_slip_mm": slip_mm,
}
(OUT_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY:", json.dumps(summary), flush=True)
