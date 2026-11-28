
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
from scipy.spatial.transform import Rotation

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT / 'outputs/nov20/ax50_brush'
OUT = P / 'continue'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene

R = Rotation.from_euler('x', 90, degrees=True)
Q = R.as_quat()
MOUNT = np.array([0., 0., .14])


def prepare():
    path = OUT / 'opposed_axis_y_v2.npz'
    if path.exists():
        return path
    r = AX50WritingScene(side='left', wrist='locked', position=MOUNT,
                         orientation=Q, gravity=(0, 0, 0), iterations=60,
                         brush_xml=P/'tongji_scene_source.xml')
    fit = json.loads((P/'grip_candidate_axis_y2.json').read_text())
    desired = np.array([fit['native_q'][fit['joint_names'].index(n)] for n in r.action_names])
    q0 = r.state.joint_q.numpy()
    initial = q0[[r.joint_names.index(n) for n in r.action_names]]
    wall = time.time()
    q = r.state.joint_q.numpy().copy()
    q[r.brush_q_start:r.brush_q_start+3] = [2, 2, -2]
    q[r.brush_hinge_q] = 0
    r.state.joint_q.assign(q)
    import newton
    newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
    r.rigid.reset(r.state, flags=0)
    r.vbd.reset(r.state, flags=0)
    r.collision.reset_contact_matching()
    for key, value in vars(r.state).items():
        dst = getattr(r.next, key, None)
        if isinstance(value, wp.array) and isinstance(dst, wp.array):
            wp.copy(dst, value)
    for k in range(80):
        u = min(1., (k + 1) / 55)
        r.set_targets(initial + (desired - initial) * (u*u*(3-2*u)))
        for _ in range(40):
            r.step(.00025)
        if k % 10 == 0:
            print('PREPARE', k, r.time, time.time()-wall, flush=True)
        if not np.isfinite(r.state.particle_q.numpy()).all():
            raise RuntimeError('nonfinite empty-hand preparation')
    names = list(r.action_names)+['tissue']
    ids = [r.joint_names.index(n) for n in names]
    np.savez(path, joint_q=r.state.joint_q.numpy()[ids], joint_qd=r.state.joint_qd.numpy(),
             particle_q=r.state.particle_q.numpy(), particle_qd=r.state.particle_qd.numpy(),
             body_q=r.state.body_q.numpy(), body_qd=r.state.body_qd.numpy(),
             frames=r.state.body_q.numpy()[None], body_names=r.body_names_all,
             action_names=r.action_names, target=r.target.numpy()[ids])
    error = np.max(np.abs(r.observe()['joint_position'] - desired))
    (OUT/'opposed_axis_y_prepare.json').write_text(json.dumps({
        'joint_error_max_rad': float(error), 'orientation_xyzw': Q.tolist(),
        'status': 'empty-hand preparation only', 'candidate': fit}, indent=2))
    print('PREPARED', error, flush=True)
    return path


def contacts(r, dt):
    labels = r.model.shape_label
    c = r.contacts
    n = min(int(c.soft_contact_count.numpy()[0]), 131072)
    shapes = c.soft_contact_shape.numpy()[:n]
    forces = r.out_f.numpy()[:n]
    depth = r.out_depth.numpy()[:n]
    indices = c.soft_contact_indices.numpy()[:n]
    bary = c.soft_contact_barycentric.numpy()[:n]
    result = []
    for i in np.flatnonzero((shapes >= 0) & (shapes < len(labels))):
        si = shapes[i]
        if labels[si] != 'brush_shaft':
            continue
        vertices = indices[i]
        for j in np.flatnonzero((vertices >= r.pad_offset) & (vertices < r.bridge_offset) & (bary[i] > 0)):
            pi = vertices[j]
            result.append({'particle': int(pi), 'weight': float(bary[i,j]),
                           'force': (forces[i]*bary[i,j]).tolist(),
                           'depth_mm': float(depth[i]*1000),
                           'normal': c.soft_contact_normal.numpy()[i].tolist()})
    c = r.rigid_observation
    n = int(c.rigid_contact_count.numpy()[0])
    a, b = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    rigid = []
    for i in range(n):
        if a[i] < 0 or b[i] < 0 or 'brush_shaft' not in (labels[a[i]], labels[b[i]]):
            continue
        rigid.append({'shapes': [labels[a[i]], labels[b[i]]],
                      'normal': c.rigid_contact_normal.numpy()[i].tolist(),
                      'force': c.force.numpy()[i].tolist(),
                      'margin0_mm': float(c.rigid_contact_margin0.numpy()[i]*1000),
                      'margin1_mm': float(c.rigid_contact_margin1.numpy()[i]*1000)})
    return result, rigid


def main():
    dt = float(os.getenv('DT', '.00005'))
    seconds = float(os.getenv('SECONDS', '.0005'))
    settle_seconds = float(os.getenv('SETTLE_SECONDS', '0'))
    local = np.array([float(os.getenv('GRIP_X', '.03356')),
                      float(os.getenv('GRIP_Y', '.048')),
                      float(os.getenv('GRIP_Z', '-.00497'))])
    free_flight = os.getenv('FREE_FLIGHT', '0') == '1'
    if free_flight:
        local = np.array([.5, .5, 0.])
    name = os.getenv('NAME', 'free_flight' if free_flight else 'opposed')
    overrides = json.loads(os.getenv('JOINT_TARGETS', '{}'))
    R = Rotation.from_euler('x', float(os.getenv('HAND_PITCH_DEG', '90')), degrees=True)
    Q = R.as_quat()
    r = AX50WritingScene(side='left', wrist='locked', position=MOUNT, orientation=Q,
                         gravity=(0, 0, float(os.getenv('GRAVITY', '-9.81'))),
                         iterations=int(os.getenv('ITERATIONS', '100')),
                         brush_xml=P/'tongji_scene_source.xml')
    prepared = np.load(prepare())
    r.restore_prepared_hand(prepared)
    q = r.state.joint_q.numpy()
    pos = R.apply(local) + MOUNT
    start = r.brush_q_start
    brush_orientation = Rotation.identity()
    if os.getenv('BRUSH_ROLL_DEG'):
        brush_orientation = brush_orientation * Rotation.from_euler('z', float(os.environ['BRUSH_ROLL_DEG']), degrees=True)
    q[start:start+7] = np.r_[pos, brush_orientation.as_quat()]
    q[r.brush_hinge_q] = 0
    r.state.joint_q.assign(q)
    import newton
    newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
    r.rigid.reset(r.state, flags=0)
    r.vbd.reset(r.state, flags=0)
    r.collision.reset_contact_matching()
    for key, value in vars(r.state).items():
        dst = getattr(r.next, key, None)
        if isinstance(value, wp.array) and isinstance(dst, wp.array):
            wp.copy(dst, value)
    initial_target = r.target.numpy().copy()
    final_target = initial_target.copy()
    for joint, value in overrides.items():
        final_target[r.joint_names.index(joint)] = value
    r.target.assign(final_target if settle_seconds == 0 else initial_target)
    origin = pos.copy()
    rows = []
    frames = [r.state.body_q.numpy()]
    particles = [r.state.particle_q.numpy()]
    wall = time.time()
    settle_steps = round(settle_seconds/dt)
    steps = settle_steps + round(seconds/dt)
    interval = max(1, round(float(os.getenv('SAMPLE_INTERVAL', '.0001'))/dt))
    for k in range(steps+1):
        if k:
            if k <= settle_steps:
                u = k/settle_steps
                target = r.target.numpy()
                ids = np.array([r.joint_names.index(j) for j in overrides])
                target[ids] = initial_target[ids] + (final_target[ids]-initial_target[ids])*(u*u*(3-2*u))
                r.target.assign(target)
            r.step(dt)
        body = r.state.body_q.numpy()
        particle = r.state.particle_q.numpy()
        finite = bool(np.isfinite(body).all() and np.isfinite(particle).all() and
                      np.isfinite(r.state.joint_q.numpy()).all() and np.isfinite(r.state.joint_qd.numpy()).all())
        if k == 0 or k % interval == 0 or k == steps or k == settle_steps or not finite:
            soft, rigid = contacts(r, dt)
            sf = np.sum([x['force'] for x in soft], axis=0).tolist() if soft else [0,0,0]
            rf = np.sum([x['force'][:3] for x in rigid], axis=0).tolist() if rigid else [0,0,0]
            c = r.rigid_observation
            nc = int(c.rigid_contact_count.numpy()[0])
            pair_a = c.rigid_contact_shape0.numpy()[:nc]
            pair_b = c.rigid_contact_shape1.numpy()[:nc]
            pairs = {}
            for a,b in zip(pair_a,pair_b):
                if a >= 0 and b >= 0:
                    pair = sorted((r.model.shape_label[a], r.model.shape_label[b]))
                    key = '{} | {}'.format(*pair)
                    pairs[key] = pairs.get(key, 0)+1
            soft_bones = {}
            pad_bones = np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
            for contact in soft:
                particle_index = contact['particle']-r.pad_offset
                bone = str(pad_bones[particle_index]) if 0 <= particle_index < len(pad_bones) else 'unmapped_soft_particle'
                entry = soft_bones.setdefault(bone, {'count': 0, 'force_sum_N': [0., 0., 0.]})
                entry['count'] += 1
                entry['force_sum_N'] = (np.asarray(entry['force_sum_N'])+contact['force']).tolist()
            rotation = Rotation.from_quat(body[r.brush_root_body,3:])
            row = {'t': r.time, 'settled': k >= settle_steps, 'finite': finite,
                   'brush_disp_mm': float(np.linalg.norm(body[r.brush_root_body,:3]-origin)*1000),
                   'brush_rotation_deg': float((rotation*brush_orientation.inv()).magnitude()*180/np.pi),
                   'brush_q': body[r.brush_root_body].tolist(),
                   'soft_shaft_count': len(soft), 'rigid_shaft_count': len(rigid),
                   'soft_force_sum_N': sf, 'rigid_force_sum_N': rf,
                   'soft_bone_summary': soft_bones,
                   'rigid_contact_pairs': pairs,
                   'max_soft_depth_mm': max([x['depth_mm'] for x in soft], default=0),
                   'soft_shaft': soft[:100], 'rigid_shaft': rigid[:30]}
            if k == 0:
                pad_geometry = {}
                axis = body[r.brush_root_body,:3]
                pad_bones = np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
                for bone in np.unique(pad_bones):
                    selection = np.flatnonzero(pad_bones == bone) + r.pad_offset
                    pp = particle[selection]
                    relative = pp - axis
                    radial = np.linalg.norm(relative[:,[0,2]], axis=1)
                    valid = (relative[:,1] > -.095) & (relative[:,1] < .035)
                    if np.any(valid):
                        j = np.flatnonzero(valid)[np.argmin(radial[valid])]
                        pad_geometry[str(bone)] = {'min_surface_gap_mm': float((radial[j]-.005)*1000),
                                                   'closest_point': pp[j].tolist(),
                                                   'local_closest_point': R.inv().apply(pp[j]-MOUNT).tolist()}
                row['pad_geometry'] = pad_geometry
            rows.append(row)
            print('SAMPLE', json.dumps({x:y for x,y in row.items() if x not in ('soft_shaft','rigid_shaft','brush_q')}), flush=True)
        if k % max(1, round(.005/dt)) == 0:
            frames.append(body)
            particles.append(particle)
        if not finite or (rows and abs(rows[-1]['brush_disp_mm']) > float(os.getenv('MAX_DISP_MM', '40'))):
            break
    stamp = name.replace('+', 'p').replace('-', 'm').replace('.', 'p')
    names = list(r.action_names)+['tissue']
    ids = np.array([r.joint_names.index(n) for n in names])
    q = r.state.joint_q.numpy()
    np.savez(OUT/f'{stamp}.npz', body_q=np.array(frames), particle_q=np.array(particles),
             body_names=r.body_names_all, state_particle_q=r.state.particle_q.numpy(),
             state_particle_qd=r.state.particle_qd.numpy(), joint_q_names=ids,
             joint_q=q[ids], state_joint_q=q, state_joint_qd=r.state.joint_qd.numpy(),
             state_body_q=r.state.body_q.numpy(), state_body_qd=r.state.body_qd.numpy(),
             joint_names=r.joint_names, bristle_q=q[r.brush_hinge_q],
             brush_q=q[r.brush_q_start:r.brush_q_start+7],
             target_names=ids, target=r.target.numpy()[ids], state_target=r.target.numpy())
    result = {'name': name, 'dt': dt, 'iterations': int(os.getenv('ITERATIONS', '100')), 'gravity': float(os.getenv('GRAVITY', '-9.81')),
              'grip_local_m': local.tolist(), 'free_flight': free_flight,
              'settle_seconds': settle_seconds, 'joint_targets': overrides,
              'initial_brush_root': pos.tolist(), 'wall_seconds': time.time()-wall, 'rows': rows,
              'status': 'diagnostic only; requires continuous 2s stability'}
    (OUT/f'{stamp}.json').write_text(json.dumps(result, indent=2))
    print('WROTE', stamp, flush=True)


if __name__ == '__main__':
    main()
