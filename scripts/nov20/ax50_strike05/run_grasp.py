
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from strike_runtime import *
import csv
import traceback

TAG = os.getenv('RUN_TAG', 'grasp_v07')
DURATION = float(os.getenv('DURATION', '3.5'))
DT = float(os.getenv('SIM_DT', '.000125'))
FINGERS = np.array(PLAN.get('fingers', [0., .8, 1., .8, -.002189346254660246, .14610134140008357, 1.4, 1., .0021927640810407495, .153072812763005, 1.4, 1., 0., .6, 1.4, 1.]), dtype=float)
THUMB = np.array(PLAN.get('thumb', [.25, .7, .55, .65, .55]), dtype=float)
TRACE_STRIDE = int(os.getenv('TRACE_STRIDE', '1'))
TRAJECTORY_STRIDE = int(os.getenv('TRAJECTORY_STRIDE', '400'))
INITIAL_CLOSED = os.getenv('INITIAL_CLOSED', '1' if PLAN else '0') == '1'
COLUMNS = ['time_s', 'phase', 'hand_stone_N', 'hand_wood_N', 'hand_total_N', 'thumb_N', 'index_N', 'middle_N', 'ring_N', 'little_N', 'palm_N', 'stone_target_N', 'wood_target_N', 'hand_target_N', 'external_support_N', 'max_penetration_m', 'max_soft_indentation_m', 'hammer_position_xyz', 'quat_xyzw', 'rel_translation_m', 'rel_rotation_deg', 'contact_links', 'active_fingers', 'grasp_ok', 'current_hold_s', 'best_hold_s', 'finite', 'dt']


def smooth(t):
    u = np.clip(t, 0., 1.)
    return u * u * (3. - 2. * u)


def save_trajectory(path, trajectory, h):
    np.savez_compressed(path, time=[r[0] for r in trajectory], body_q=[r[1] for r in trajectory],
                        body_names=h.body_names_all, particles=[r[2] for r in trajectory],
                        tri=h.model.tri_indices.numpy(), targets=[r[3] for r in trajectory])


def summarize(h, bq, metrics, hammer, reference, current, best, finite, phase, dt):
    active = metrics['active']
    relative = angle = float('nan')
    if reference is not None:
        _, local_position, local_rotation = relative_hammer(h, bq)
        relative = float(np.linalg.norm(local_position - reference[0]))
        angle = float(np.rad2deg((local_rotation * reference[1].inv()).magnitude()))
    parts = metrics['parts']
    grasp_ok = bool(finite and metrics['hand_total'] > .02 and metrics['support'] < .01 and
                    relative < .005 and angle < 45. and len(active) >= 2 and hammer[2] > .10)
    current = current + dt if grasp_ok else 0.
    best = max(best, current)
    row = dict(time_s=h.time, phase=phase, hand_stone_N=parts['stone'], hand_wood_N=parts['wood'],
               hand_total_N=metrics['hand_total'], **{f'{f}_N': n for f, n in metrics['finger_n'].items()},
               palm_N=metrics['palm'], stone_target_N=parts['stone_target'], wood_target_N=parts['wood_target'],
               hand_target_N=parts['hand_target'], external_support_N=metrics['support'],
               max_penetration_m=metrics['rigid_penetration'], max_soft_indentation_m=metrics['soft_indentation'],
               hammer_position_xyz=' '.join(map(str, hammer[:3])), quat_xyzw=' '.join(map(str, hammer[3:])),
               rel_translation_m=relative, rel_rotation_deg=angle, contact_links=';'.join(sorted(metrics['links'])),
               active_fingers=';'.join(active), grasp_ok=grasp_ok, current_hold_s=current, best_hold_s=best,
               finite=finite, dt=DT)
    return row, current, best


def final_result(result, traces, stop, error):
    if not traces:
        return dict(result, status=stop, error=error, completed_steps=0, grasp_acceptance=False)
    last = traces[-1]
    return dict(result, status=stop, error=error, time_s=last['time_s'], completed_steps=len(traces),
                best_continuous_hold_s=last['best_hold_s'], hammer_position=[float(x) for x in last['hammer_position_xyz'].split()] + [float(x) for x in last['quat_xyzw'].split()],
                hand_total_N=last['hand_total_N'], external_support_N=last['external_support_N'], active_fingers=last['active_fingers'].split(';') if last['active_fingers'] else [],
                final_palm_relative_translation_m=last['rel_translation_m'], final_palm_relative_rotation_deg=last['rel_rotation_deg'],
                max_penetration_m=max(r['max_penetration_m'] for r in traces), max_soft_indentation_m=max(r['max_soft_indentation_m'] for r in traces),
                stone_target_peak_N=max(r['stone_target_N'] for r in traces), hand_target_peak_N=max(r['hand_target_N'] for r in traces),
                external_support_peak_N=max(r['external_support_N'] for r in traces), finite=all(r['finite'] for r in traces),
                grasp_acceptance=bool(stop == 'completed' and last['best_hold_s'] >= 2.))


def run():
    h = construct()
    meta = static_check(h)
    initial_target = hand_targets(h.action_names, FINGERS, THUMB)
    if INITIAL_CLOSED:
        old_bq = h.state.body_q.numpy().copy()
        old_p = h.state.particle_q.numpy().copy()
        q = h.state.joint_q.numpy()
        for i, dof in enumerate(h.action_indices):
            if dof >= 6:
                q[dof] = initial_target[i]
        for state in [h.state, h.next]:
            state.joint_q.assign(q)
            state.joint_qd.assign(np.zeros_like(state.joint_qd.numpy()))
        newton.eval_fk(h.model, h.state.joint_q, h.state.joint_qd, h.state)
        newton.eval_fk(h.model, h.next.joint_q, h.next.joint_qd, h.next)
        new_bq = h.state.body_q.numpy()
        owners = h.attachment_owner.numpy()
        moved = old_p.copy()
        for b in np.unique(owners):
            if b < 0 or b >= len(new_bq): continue
            ids = np.flatnonzero(owners == b)
            ids = ids[ids < len(old_p)]
            local = Rotation.from_quat(old_bq[b,3:]).inv().apply(old_p[ids]-old_bq[b,:3])
            moved[ids] = Rotation.from_quat(new_bq[b,3:]).apply(local)+new_bq[b,:3]
        for state in [h.state,h.next]:
            state.particle_q.assign(moved)
            state.particle_qd.zero_()
        h.feedback.zero_()
    initial_recs = h.sample_contact()
    initial_metrics = evaluate_contacts(h, initial_recs)
    meta.update(tag=TAG, duration_s=DURATION, dt_s=DT, finger_target=FINGERS.tolist(), thumb_target=THUMB.tolist(),
                named_joint_targets={name: float(initial_target[i]) for i, name in enumerate(h.action_names) if abs(initial_target[i]) > 0},
                grasp_plan=PLAN_PATH or None, plan_sha256=sha256(PLAN_PATH) if PLAN_PATH else None, initial_closed_pose=INITIAL_CLOSED,
                drift_translation_limit_m=.005, drift_rotation_limit_deg=45., per_finger_threshold_N=.01,
                initial_rigid_penetration_m=initial_metrics['rigid_penetration'], initial_soft_indentation_m=initial_metrics['soft_indentation'],
                trajectory_stride=TRAJECTORY_STRIDE, metric_stride=TRACE_STRIDE,
                scope='actual Newton per-substep hand PD grasp of the original free hammer; no hammer target, force, attachment, or selective reset; not strike acceptance')
    (OUT / f'{TAG}_model.json').write_text(json.dumps(meta, indent=2))
    traces, contacts, trajectory = [], [], []
    current = best = 0.
    reference = None
    stop = 'running'
    error = None
    result = dict(tag=TAG, scope=meta['scope'], status='running', time_s=h.time, grasp_acceptance=False)
    (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
    try:
        if initial_metrics['rigid_penetration'] > .0005:
            stop = 'initial_rigid_penetration_guard_stop'
        elif not initial_metrics['finite']:
            stop = 'initial_nonfinite_contact'
        for step in range(round(DURATION / DT)) if stop == 'running' else []:
            t = step * DT
            phase = 'fitted_grasp_prepose_hold' if INITIAL_CLOSED else ('close_fingers' if t < .4 else 'close_thumb' if t < .9 else 'hold')
            f = th = 1. if INITIAL_CLOSED else (smooth(t / .4) if t < .4 else 1.)
            if not INITIAL_CLOSED:
                th = 0. if t < .4 else smooth((t - .4) / .5) if t < .9 else 1.
            target = hand_targets(h.action_names, FINGERS * f, THUMB * th)
            h.set_targets(target)
            h.step(DT)
            bq = h.state.body_q.numpy()
            hammer = bq[h.hammer_body].copy()
            recs = h.sample_contact()
            metrics = evaluate_contacts(h, recs)
            finite = metrics['finite']
            if step % TRACE_STRIDE == 0 or metrics['rigid_penetration'] > .0005:
                finite = finite and all(np.isfinite(getattr(h.state, key).numpy()).all() for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd'])
                if reference is None and h.time >= .05 and len(metrics['active']) >= 2 and metrics['support'] < .01:
                    _, position, rotation = relative_hammer(h, bq)
                    reference = (position.copy(), rotation)
                row, current, best = summarize(h, bq, metrics, hammer, reference, current, best, finite, phase, DT * (TRACE_STRIDE if step >= TRACE_STRIDE else 1))
                traces.append(row)
                for r in recs:
                    if r['fn'] > 1.e-9 or r['depth'] > 0:
                        contacts.append([h.time, r['kind'], r['hand'], r['particle'], r['object'], r['fn'], r['depth'], r['target'], r['support']])
                if step % (TRACE_STRIDE * 40) == 0:
                    result = final_result(result, traces, 'running', None)
                    (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
                    with (OUT / f'{TAG}_trace.csv').open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(traces)
                    print('GRASP', json.dumps({k: result[k] for k in ['time_s', 'best_continuous_hold_s', 'hand_total_N', 'external_support_N', 'max_penetration_m', 'active_fingers']}), flush=True)
            if step % TRAJECTORY_STRIDE == 0:
                trajectory.append((h.time, bq.copy(), h.state.particle_q.numpy()[:h.real_count].copy(), target.copy()))
                save_trajectory(OUT / f'{TAG}_live.npz', trajectory, h)
                np.savez_compressed(OUT / f'{TAG}_checkpoint.npz', **h.get_checkpoint())
            if metrics['rigid_penetration'] > .0005:
                stop = 'rigid_penetration_guard_stop'
                break
            if not finite:
                stop = 'nonfinite_state_or_contact'
                break
            if h.time >= .25 and metrics['hand_total'] <= .02 and current == 0 and reference is None:
                stop = 'early_no_grasp_contact'
                break
            if hammer[2] < .10:
                stop = 'early_hammer_fell'
                break
        if stop == 'running':
            stop = 'completed' if DURATION >= 3.5 else 'short_real_smoke_not_acceptance'
    except Exception as e:
        stop = 'software_or_simulation_error'
        error = repr(e)
        print(traceback.format_exc(), flush=True)
    finally:
        if traces:
            with (OUT / f'{TAG}_trace.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(traces)
            with (OUT / f'{TAG}_contacts.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['time_s', 'type', 'hand_part', 'particle', 'object', 'normal_N', 'penetration_m', 'target', 'support']); w.writerows(contacts)
            if not trajectory or trajectory[-1][0] < h.time:
                trajectory.append((h.time, h.state.body_q.numpy().copy(), h.state.particle_q.numpy()[:h.real_count].copy(), target.copy()))
            save_trajectory(OUT / f'{TAG}.npz', trajectory, h)
            np.savez_compressed(OUT / f'{TAG}_checkpoint.npz', **h.get_checkpoint())
        result = final_result(result, traces, stop, error)
        (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
        (OUT / f'{TAG}_status.json').write_text(json.dumps(dict(tag=TAG, status=stop, error=error, time_s=h.time), indent=2))
        print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
