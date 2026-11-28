
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json, os, sys, time
import numpy as np
import warp as wp

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT / 'outputs/nov20/ax50_brush'
OUT = P / 'continue'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene


def contact_snapshot(r):
    c = r.rigid_observation
    n = int(c.rigid_contact_count.numpy()[0])
    a = c.rigid_contact_shape0.numpy()[:n]
    b = c.rigid_contact_shape1.numpy()[:n]
    force = c.force.numpy()[:n]
    labels = r.model.shape_label
    p0 = c.rigid_contact_point0.numpy()[:n]
    p1 = c.rigid_contact_point1.numpy()[:n]
    normal = c.rigid_contact_normal.numpy()[:n]
    separation = np.linalg.norm(p1 - p0, axis=1)
    rigid = []
    for i in range(n):
        if a[i] < 0 or b[i] < 0 or 'brush_shaft' not in (labels[a[i]], labels[b[i]]):
            continue
        rigid.append({
            'shapes': [labels[a[i]], labels[b[i]]],
            'point0': p0[i].tolist(), 'point1': p1[i].tolist(),
            'normal': normal[i].tolist(), 'point_separation_mm': float(separation[i] * 1000),
            'force': force[i].tolist(),
        })
    c = r.contacts
    n = int(c.soft_contact_count.numpy()[0])
    shapes = c.soft_contact_shape.numpy()[:n]
    select = np.flatnonzero(np.array([0 <= i < len(labels) and labels[i] == 'brush_shaft' for i in shapes]))
    soft = []
    for i in select:
        soft.append({
            'vertices': c.soft_contact_indices.numpy()[i].tolist(),
            'barycentric': c.soft_contact_particle.numpy()[i].tolist(),
            'point': c.soft_contact_body_pos.numpy()[i].tolist(),
            'normal': c.soft_contact_normal.numpy()[i].tolist(),
            'force': r.out_f.numpy()[i].tolist(),
            'depth_mm': float(r.out_depth.numpy()[i] * 1000),
        })
    return {'rigid_shaft': rigid, 'soft_shaft': soft}


def main():
    dt = float(os.getenv('DT', '.0001'))
    seconds = float(os.getenv('SECONDS', '.02'))
    iterations = int(os.getenv('ITERATIONS', '60'))
    candidate = os.getenv('CANDIDATE', 'compact')
    if candidate != 'compact':
        raise ValueError('first diagnostic uses the original compact contact case')
    r = AX50WritingScene(side='left', wrist='locked', position=(0, 0, .14),
                         gravity=(0, 0, -9.81), iterations=iterations,
                         brush_xml=P/'tongji_scene_source.xml')
    r.restore_prepared_hand(np.load(P/'prepared_hand_locked.npz'))
    original = r.target.numpy().copy()
    closed = original.copy()
    closed[r.joint_names.index('digit_2_flex_1')] -= float(os.getenv('INDEX_CLOSE', '.05'))
    closed[r.joint_names.index('digit_3_flex_1')] -= float(os.getenv('MIDDLE_CLOSE', '.04'))
    start = r.state.body_q.numpy()[r.brush_root_body].copy()
    wall = time.time()
    rows = []
    steps = round(seconds / dt)
    for k in range(steps + 1):
        u = min(1.0, k * dt / .15)
        r.target.assign(original + (closed-original) * u)
        if k:
            r.step(dt)
        if k == 0 or k % round(.001/dt) == 0 or k == steps:
            body = r.state.body_q.numpy()
            particles = r.state.particle_q.numpy()
            snap = contact_snapshot(r)
            rigid_forces = np.array([x['force'][:3] for x in snap['rigid_shaft']])
            soft_forces = np.array([x['force'] for x in snap['soft_shaft']])
            row = {
                'step': k, 't': r.time,
                'brush_disp_mm': float(np.linalg.norm(body[r.brush_root_body,:3]-start[:3])*1000),
                'brush_quat_change': float(np.linalg.norm(body[r.brush_root_body,3:]-start[3:])),
                'rigid_contacts': len(snap['rigid_shaft']),
                'soft_contacts': len(snap['soft_shaft']),
                'rigid_force_sum_N': rigid_forces.sum(0).tolist() if len(rigid_forces) else [0,0,0],
                'soft_force_sum_N': soft_forces.sum(0).tolist() if len(soft_forces) else [0,0,0],
                'finite': bool(np.isfinite(body).all() and np.isfinite(particles).all()),
                'contacts': snap,
            }
            rows.append(row)
            print(json.dumps({x:y for x,y in row.items() if x != 'contacts'}), flush=True)
            print('RIGID_NORMALS', json.dumps(snap['rigid_shaft'][:6]), flush=True)
            print('SOFT_NORMALS', json.dumps(snap['soft_shaft'][:6]), flush=True)
            if not row['finite']:
                break
    result = {'dt': dt, 'iterations': iterations, 'seconds_requested': seconds,
              'wall_seconds': time.time()-wall, 'rows': rows,
              'status': 'physical contact diagnostic; not an accepted grasp'}
    (OUT/'contact_diagnostic.json').write_text(json.dumps(result, indent=2))
    print('WROTE', OUT/'contact_diagnostic.json', flush=True)


if __name__ == '__main__':
    main()
