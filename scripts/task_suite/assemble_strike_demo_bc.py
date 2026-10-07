"""Quality-gate and combine physically successful native-scene Strike action demos."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--reports', type=Path, nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--min_traces', type=int, default=2)
args = parser.parse_args()
obs_parts, action_parts, manifest, hashes = [], [], [], set()
for report_path in args.reports:
    report = json.loads(report_path.read_text())
    if report.get('task') != 'strike' or report.get('curriculum_stage') != 'stage2':
        raise ValueError(f'Wrong task or stage: {report_path}')
    if report.get('morphology') != '13_14' or not report.get('training_scene'):
        raise ValueError(f'Wrong morphology or changed scene: {report_path}')
    if report.get('controller') != 'strike_joint_target_v1' or not report.get('success'):
        raise ValueError(f'No valid direct-action full success: {report_path}')
    overshoot = float(report.get('max_joint_limit_violation_rad', 1e9))
    if report.get('joint_limit_violation_steps') != 0 or overshoot > 0.02:
        raise ValueError(f'Joint-limit quality gate failed: {report_path}')
    if not any(float(step.get('reward', 0)) >= 1000 for step in report['history']):
        raise ValueError(f'No sparse full-task terminal reward: {report_path}')
    trace_path = Path(report['scripted_trace'])
    if not trace_path.is_absolute():
        trace_path = report_path.parents[3] / trace_path
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if digest in hashes:
        raise ValueError(f'Duplicate action trace: {trace_path}')
    hashes.add(digest)
    with np.load(trace_path) as trace:
        obs = np.asarray(trace['observations_before_step'], dtype=np.float32)
        actions = np.asarray(trace['submitted_actions'], dtype=np.float32)
        if obs.shape != (report['steps_executed'], 162) or actions.shape != (report['steps_executed'], 23):
            raise ValueError(f'Wrong BC tensor shape: {trace_path}')
        if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001:
            raise ValueError(f'Invalid BC tensor values: {trace_path}')
        if not bool(np.asarray(trace['actions_control_fingers']).all()) or not bool(np.asarray(trace['scene_unmodified']).all()):
            raise ValueError(f'Action or scene provenance failed: {trace_path}')
    obs_parts.append(obs)
    action_parts.append(actions)
    manifest.append({'report': str(report_path), 'trace': str(trace_path), 'sha256': digest,
                     'samples': len(obs), 'max_joint_limit_violation_rad': overshoot})
if len(manifest) < args.min_traces:
    raise ValueError(f'Need at least {args.min_traces} distinct successful action traces')
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output,
    observations_before_step=np.concatenate(obs_parts),
    submitted_actions=np.concatenate(action_parts),
    actions_control_fingers=np.ones(sum(map(len, obs_parts)), dtype=bool),
    scene_unmodified=np.ones(sum(map(len, obs_parts)), dtype=bool))
args.output.with_suffix('.manifest.json').write_text(json.dumps({'task': 'strike', 'stage': 'stage2',
    'morphology': '13_14', 'controller': 'strike_joint_target_v1', 'samples': sum(map(len, obs_parts)),
    'sources': manifest}, indent=2))
print(json.dumps({'traces': len(manifest), 'samples': sum(map(len, obs_parts)), 'dataset': str(args.output)}))
