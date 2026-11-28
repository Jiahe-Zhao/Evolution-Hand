"""Run AX50 latch-16 (hand positive & direct negative) in Newton MuJoCo-Warp + VBD."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import csv
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from latch_runtime import *

mode = sys.argv[1] if len(sys.argv) > 1 else 'hand'
assert mode in ('hand', 'direct'), f'Unknown mode: {mode}'

# Mount: local +X -> world -X (fingers point -X), local +Y -> world -Z (palm down), local +Z -> world -Y
R_MOUNT = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0],
], dtype=float)
QUAT = R.from_matrix(R_MOUNT).as_quat().tolist()

# Initial neutral index_tip offset in world is R_MOUNT @ [0.1111, 0, -0.0277] = [-0.1111, +0.0277, 0]
INIT_TIP = np.array([-0.128, 0.001, 0.130], dtype=float)
MOUNT_POS = (INIT_TIP - np.array([-0.1111, -0.0277, 0.0])).tolist()

h = LatchRuntime(
    side='left',
    wrist='six_axis',
    position=MOUNT_POS,
    orientation=QUAT,
    gravity=(0, 0, -9.81),
    iterations=50,
)
h.vbd.prop_contact_ke = 500.0
h.vbd.prop_contact_kd = 0.05
(OUT_DIR / f'{mode}_model.json').write_text(json.dumps(h.latch_info(), indent=2))

# Check actual initial link_2_3 position
idx_tip_body = h.body_names.index('link_2_3')
init_tip_actual = h.state.body_q.numpy()[idx_tip_body, :3].copy()
print(f'[{mode}] init_tip_actual={init_tip_actual.round(4)}', flush=True)

# Joint indices: 0..5 wrist, 6..9 index (arch_2, flex_1..3), 10..13 middle, 14..17 ring, 18..21 pinky, 22..26 thumb
n_dof = len(h.action_names)
preshape = np.zeros(n_dof, dtype=float)
# Tuck middle, ring, pinky out of the way; keep index straight
for base_idx in (10, 14, 18):
    preshape[base_idx + 1] = 0.45
    preshape[base_idx + 2] = 0.45
    preshape[base_idx + 3] = 0.25

def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

# Build waypoints (time_s, world_tip_pos, phase)
if mode == 'hand':
    pts = [
        (0.00, init_tip_actual, 'preshape'),
        (0.40, init_tip_actual, 'approach_bolt'),
        (0.80, np.array([-0.167, 0.020, 0.112]), 'lower_to_bolt'),
        (1.10, np.array([-0.167, 0.020, 0.094]), 'push_bolt'),
        (1.65, np.array([-0.167, 0.038, 0.094]), 'retract_bolt'),
        (1.95, np.array([-0.133, 0.038, 0.096]), 'move_to_pull'),
        (2.25, np.array([-0.133, -0.002, 0.091]), 'enter_pull'),
        (2.55, np.array([-0.153, -0.002, 0.092]), 'pull_lid'),
    ]
    t0 = 2.55
    for k, ang in enumerate(np.linspace(0.0, 78.0, 11)):
        if k == 0:
            continue
        th = np.deg2rad(ang)
        pos = np.array([
            -0.202 + 0.049 * np.cos(th) - 0.019 * np.sin(th),
            -0.002,
            0.078 + 0.049 * np.sin(th) + 0.020 * np.cos(th),
        ])
        pts.append((t0 + 0.145 * k, pos, 'pull_lid'))
    total_time = 4.10
else:
    pts = [
        (0.00, init_tip_actual, 'preshape'),
        (0.40, init_tip_actual, 'move_to_pull'),
        (0.90, np.array([-0.133, -0.002, 0.091]), 'enter_pull'),
        (1.30, np.array([-0.153, -0.002, 0.092]), 'pull_lid'),
    ]
    t0 = 1.30
    for k, ang in enumerate(np.linspace(0.0, 22.0, 6)):
        if k == 0:
            continue
        th = np.deg2rad(ang)
        pos = np.array([
            -0.202 + 0.049 * np.cos(th) - 0.019 * np.sin(th),
            -0.002,
            0.078 + 0.049 * np.sin(th) + 0.020 * np.cos(th),
        ])
        pts.append((t0 + 0.15 * k, pos, 'pull_lid'))
    total_time = 2.15

sb = h.model.shape_body.numpy()
shape_labels = [str(x).rsplit('/', 1)[-1] for x in h.model.shape_label]

rows = []
contacts_log = []
body_traj = []
particle_traj = []
n_steps = int(round(total_time / 0.01))
stop_reason = 'completed'

for step_i in range(n_steps):
    t = step_i * 0.01
    target = preshape.copy()
    if t < 0.40:
        target *= smooth(t / 0.40)
        dst = init_tip_actual
        phase = 'preshape'
    else:
        dst = pts[-1][1]
        phase = pts[-1][2]
        for (ta, pa, ph), (tb, pb, _) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                u = smooth((t - ta) / max(tb - ta, 1e-6))
                dst = pa + (pb - pa) * u
                phase = ph
                break
    d_world = dst - init_tip_actual
    target[0:3] = R_MOUNT.T @ d_world

    if mode == 'hand' and t >= 2.55:
        u_hook = smooth((t - 2.55) / 1.45)
        target[7] = 0.70 * u_hook
        target[8] = 0.55 * u_hook
        target[9] = 0.40 * u_hook
        d_world = d_world + np.array([-0.062 * u_hook, 0.0, -0.018 * u_hook])
        target[0:3] = R_MOUNT.T @ d_world
    h.set_targets(target)
    h.advance(0.01, dt=0.0000625 if ((mode == "hand" and t >= 2.50) or (mode == "direct" and t >= 1.25)) else 0.000125)

    q = h.state.joint_q.numpy()
    qd = h.state.joint_qd.numpy()
    pq = h.state.particle_q.numpy()
    pv = h.state.particle_qd.numpy()
    bq = h.state.body_q.numpy()
    finite = bool(all(np.isfinite(x).all() for x in (q, qd, pq, pv, bq)))
    bolt_mm = float(q[h.slide_q] * 1000.0)
    lid_deg = float(np.rad2deg(q[h.lid_q]))
    soft_v = float(np.nanmax(np.linalg.norm(pv, axis=1)))

    # Collect rigid & soft contacts
    idx_bolt_N = 0.0
    idx_lid_N = 0.0
    ear_tongue_N = 0.0
    c_rig = h.rigid_observation
    nr = int(c_rig.rigid_contact_count.numpy()[0])
    s0 = c_rig.rigid_contact_shape0.numpy()[:nr]
    s1 = c_rig.rigid_contact_shape1.numpy()[:nr]
    ff = c_rig.force.numpy()[:nr, :3]
    nm = c_rig.rigid_contact_normal.numpy()[:nr]
    for j in range(nr):
        fn = abs(float(np.dot(ff[j], nm[j])))
        if fn <= 1e-8:
            continue
        b0, b1 = int(sb[s0[j]]), int(sb[s1[j]])
        n0, n1 = shape_labels[s0[j]], shape_labels[s1[j]]
        if {n0, n1} == {'ear', 'tongue'}:
            ear_tongue_N += fn
            contacts_log.append([h.time, 'rigid', n0, n1, fn])
        for own, other, other_s in ((b0, b1, n1), (b1, b0, n0)):
            if 0 <= own < h.hand_body_count and other in (h.bolt_body, h.lid_body):
                bname = h.body_names[own]
                if other == h.bolt_body and bname == 'link_2_3':
                    idx_bolt_N += fn
                elif other == h.lid_body and bname == 'link_2_3':
                    idx_lid_N += fn
                contacts_log.append([h.time, 'rigid', bname, other_s, fn])

    c_soft = h.contacts
    ns = int(c_soft.soft_contact_count.numpy()[0])
    sh = c_soft.soft_contact_shape.numpy()[:ns]
    ii = c_soft.soft_contact_indices.numpy()[:ns]
    bb = c_soft.soft_contact_barycentric.numpy()[:ns]
    sff = h.out_f.numpy()[:ns]
    snm = c_soft.soft_contact_normal.numpy()[:ns]
    for j in range(ns):
        other = int(sb[sh[j]])
        if other not in (h.bolt_body, h.lid_body):
            continue
        fn = abs(float(np.dot(sff[j], snm[j])))
        if fn <= 1e-8:
            continue
        other_s = shape_labels[sh[j]]
        weights = {}
        for pid, w in zip(ii[j], bb[j]):
            if w > 0 and 0 <= pid < h.real_count:
                reg = h.region_names[h.regions[pid]] if h.regions[pid] >= 0 else 'other'
                weights[reg] = weights.get(reg, 0.0) + float(w)
        for reg, w in weights.items():
            f_part = fn * w
            if other == h.bolt_body and reg == 'index_distal':
                idx_bolt_N += f_part
            elif other == h.lid_body and reg == 'index_distal':
                idx_lid_N += f_part
            contacts_log.append([h.time, 'soft', reg, other_s, f_part])

    rows.append([h.time, phase, bolt_mm, lid_deg, idx_bolt_N, idx_lid_N, ear_tongue_N, finite, soft_v])
    body_traj.append(bq)
    particle_traj.append(pq[:h.real_count])

    if step_i % 20 == 0 or step_i == n_steps - 1:
        print(
            f'[{mode}] t={h.time:.2f}s {phase:13s} bolt={bolt_mm:5.2f}mm lid={lid_deg:6.2f}deg '
            f'idx_bolt={idx_bolt_N:.3f}N idx_lid={idx_lid_N:.3f}N ear_t={ear_tongue_N:.3f}N softv={soft_v:.3f}',
            flush=True,
        )
    if not finite or soft_v > 25.0:
        stop_reason = 'nonfinite_or_runaway'
        break

np.savez_compressed(
    OUT_DIR / f'{mode}.npz',
    time=np.array([r[0] for r in rows]),
    body_q=np.array(body_traj),
    body_names=h.body_names_all,
    particles=np.array(particle_traj),
    tri=h.model.tri_indices.numpy(),
    bolt_mm=np.array([r[2] for r in rows]),
    lid_deg=np.array([r[3] for r in rows]),
)
with (OUT_DIR / f'{mode}_trace.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'phase', 'bolt_mm', 'lid_deg', 'index_bolt_N', 'index_lid_N', 'ear_tongue_N', 'finite', 'max_soft_speed_m_s'])
    w.writerows(rows)
with (OUT_DIR / f'{mode}_contacts.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'type', 'hand_or_g1', 'obj_or_g2', 'normal_N'])
    w.writerows(contacts_log)

arr = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows])
summary = {
    'mode': mode,
    'stop': stop_reason,
    'time_s': float(h.time),
    'finite': bool(all(r[7] for r in rows)),
    'max_bolt_mm': float(arr[:, 0].max()),
    'max_lid_deg': float(arr[:, 1].max()),
    'end_bolt_mm': float(arr[-1, 0]),
    'end_lid_deg': float(arr[-1, 1]),
    'max_index_bolt_N': float(arr[:, 2].max()),
    'max_index_lid_N': float(arr[:, 3].max()),
    'max_ear_tongue_N': float(arr[:, 4].max()),
}
(OUT_DIR / f'{mode}_summary.json').write_text(json.dumps(summary, indent=2))
print('SUMMARY:', json.dumps(summary), flush=True)
