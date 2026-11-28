
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from strike_runtime import *
from run_grasp import smooth, save_trajectory, COLUMNS, FINGERS, THUMB
import csv
import traceback

TAG = os.getenv('RUN_TAG', 'strike_v02')
SOURCE = os.getenv('GRASP_TAG', '')
DURATION = float(os.getenv('DURATION', '3.5'))
DT = float(os.getenv('SIM_DT', '.000125'))
TRACE_STRIDE = int(os.getenv('TRACE_STRIDE', '1'))
TRAJECTORY_STRIDE = int(os.getenv('TRAJECTORY_STRIDE', '400'))
APPROACH = np.array([float(os.getenv(k, v)) for k, v in [('APPROACH_X', '0'), ('APPROACH_Y', '0'), ('APPROACH_Z', '0')]])
DOWN = float(os.getenv('STRIKE_DOWN', '.035'))
START = float(os.getenv('APPROACH_TIME', '2.'))


def restore(h, tag):
    cp = np.load(OUT / f'{tag}_checkpoint.npz')
    for state in [h.state, h.next]:
        for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd']:
            getattr(state, key).assign(cp[key])
    h.feedback.assign(cp['feedback'])
    h.target.assign(cp['target'])
    h.time = float(cp['time'])
    newton.eval_fk(h.model, h.state.joint_q, h.state.joint_qd, h.state)
    return float(cp['time'])


def finish(result, traces, stop, error):
    if not traces:
        return dict(result, status=stop, error=error, completed_steps=0, strike_acceptance=False)
    last = traces[-1]
    stone_peak = max(r['stone_target_N'] for r in traces)
    hand_peak = max(r['hand_target_N'] for r in traces)
    return dict(result, status=stop, error=error, time_s=last['time_s'], completed_steps=len(traces),
                best_continuous_preimpact_hold_s=last['best_hold_s'], stone_target_peak_N=stone_peak,
                wood_target_peak_N=max(r['wood_target_N'] for r in traces), hand_target_peak_N=hand_peak,
                hand_total_N=last['hand_total_N'], external_support_N=last['external_support_N'],
                final_palm_relative_translation_m=last['rel_translation_m'], final_palm_relative_rotation_deg=last['rel_rotation_deg'],
                active_fingers=last['active_fingers'].split(';') if last['active_fingers'] else [],
                max_penetration_m=max(r['max_penetration_m'] for r in traces),
                max_soft_indentation_m=max(r['max_soft_indentation_m'] for r in traces),
                external_support_peak_N=max(r['external_support_N'] for r in traces), finite=all(r['finite'] for r in traces),
                first_stone_target_time_s=next((r['time_s'] for r in traces if r['stone_target_N'] > 0), None),
                strike_acceptance=bool(stop == 'completed' and last['best_hold_s'] >= 2. and stone_peak >= 5. and hand_peak <= 1.e-9))


def run():
    if not SOURCE:
        raise ValueError('GRASP_TAG is required: strike starts only from a recorded real grasp')
    source_result = json.loads((OUT / f'{SOURCE}_result.json').read_text())
    if not source_result.get('grasp_acceptance') or source_result.get('status') != 'completed':
        raise ValueError(f'accepted completed grasp required, got {SOURCE}: {source_result.get("status")}, acceptance={source_result.get("grasp_acceptance")}')
    h = construct()
    meta = static_check(h)
    t0 = restore(h, SOURCE)
    meta.update(tag=TAG, scope='hand PD only, original free hammer, fixed target; no prop force, attachment or selective reset',
                source_grasp=SOURCE, source_result_sha256=sha256(OUT / f'{SOURCE}_result.json'), source_restored_world_time_s=t0,
                approach_local_m=APPROACH.tolist(), strike_down_local_m=DOWN,
                preimpact_hold_requirement_s=2., stone_target_threshold_N=5., drift_translation_limit_m=.005)
    (OUT / f'{TAG}_model.json').write_text(json.dumps(meta, indent=2))
    traces, contacts, trajectory = [], [], []
    current = best = 0.
    reference = None
    stop = 'running'
    error = None
    result = dict(tag=TAG, scope=meta['scope'], status='running', time_s=h.time, strike_acceptance=False)
    (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
    try:
        bq = h.state.body_q.numpy()
        initial = evaluate_contacts(h, h.sample_contact())
        if initial['rigid_penetration'] > .0005:
            stop = 'initial_rigid_penetration_guard_stop'
        for step in range(round(DURATION / DT)) if stop == 'running' else []:
            if step * DT >= START and best < 2.:
                stop = 'insufficient_continuous_preimpact_hold'
                break
            phase = 'verified_preimpact_hold' if step * DT < START else 'approach'
            if step * DT >= START + .5:
                phase = 'stone_strike'
            if step * DT >= START + .75:
                phase = 'recoil'
            wrist = APPROACH * smooth((step * DT - START) / .5)
            wrist[2] -= DOWN * smooth((step * DT - START - .5) / .125) * (1. - smooth((step * DT - START - .75) / .4))
            target = hand_targets(h.action_names, FINGERS, THUMB, wrist)
            h.set_targets(target)
            h.step(DT)
            bq = h.state.body_q.numpy()
            hammer = bq[h.hammer_body].copy()
            recs = h.sample_contact()
            metrics = evaluate_contacts(h, recs)
            finite = metrics['finite']
            if step % TRACE_STRIDE == 0 or metrics['rigid_penetration'] > .0005 or metrics['parts']['stone_target'] > 0:
                finite = finite and all(np.isfinite(getattr(h.state, key).numpy()).all() for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd'])
                if reference is None:
                    _, position, rotation = relative_hammer(h, bq)
                    reference = (position.copy(), rotation)
                row, proposed_current, proposed_best = __import__('run_grasp').summarize(h, bq, metrics, hammer, reference, current, best, finite, phase, DT * (TRACE_STRIDE if step >= TRACE_STRIDE else 1))
                if step * DT >= START and row['stone_target_N'] <= 0 and row['wood_target_N'] <= 0:
                    proposed_current = 0.
                    row.update(grasp_ok=False, current_hold_s=0.)
                proposed_best = max(best, proposed_current)
                row['best_hold_s'] = proposed_best
                current, best = proposed_current, proposed_best
                traces.append(row)
                for r in recs:
                    if r['fn'] > 1.e-9 or r['depth'] > 0:
                        contacts.append([h.time, r['kind'], r['hand'], r['particle'], r['object'], r['fn'], r['depth'], r['target'], r['support']])
                if step % (TRACE_STRIDE * 40) == 0 or metrics['parts']['stone_target'] > 0:
                    result = finish(result, traces, 'running', None)
                    (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
                    print('STRIKE', json.dumps({k: result[k] for k in ['time_s', 'best_continuous_preimpact_hold_s', 'stone_target_peak_N', 'hand_target_peak_N', 'finite']}), flush=True)
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
            if step * DT >= START and current < 2. and best < 2. and step * DT >= START + 1.5:
                stop = 'insufficient_continuous_preimpact_hold'
                break
        if stop == 'running':
            stop = 'completed'
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
        result = finish(result, traces, stop, error)
        (OUT / f'{TAG}_result.json').write_text(json.dumps(result, indent=2))
        (OUT / f'{TAG}_status.json').write_text(json.dumps(dict(tag=TAG, status=stop, error=error, time_s=h.time), indent=2))
        print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
