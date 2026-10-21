"""Offline geometry at recorded policy anomalies; not contact-force attribution."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

base = Path(__file__).resolve().parent
cal = base.parent / 'contact_control_calibration_20261021'
sys.path.insert(0, str(cal))
from plan_grasp import Model

audit = json.loads((cal / 'wrap04/audit.json').read_text())
model = Model(base.parent / 'motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf', face='ymixed')
plan = json.loads((cal / 'plans_wrap_cover_selected.json').read_text())[0]
G = np.array(plan['rotation']).T
root_R = G @ model.root[:3, :3]
base_pos = np.array([0, 0, .3]) + G @ (model.root[:3, 3] - plan['center'])
names = audit['joint_names']
finger_ids = [names.index(n) for n in model.names]
wrist_ids = [names.index('diagnostic_wrist_' + a) for a in 'xyz']
results = []
for path in sorted(base.glob('diagnostic_*/diagnostic_report.json')):
    report = json.loads(path.read_text())
    entries = []
    for event in report['events']:
        snapshots = {}
        for prefix in ['', 'old_']:
            q = np.array(event[prefix + 'q'])
            state = np.array(event[prefix + 'object'])
            root = np.array(event['origin']) + base_pos + root_R @ q[wrist_ids]
            center = model.root[:3, 3] + G.T @ (state[:3] - root)
            rotation = G.T @ Rotation.from_quat(state[[4, 5, 6, 3]]).as_matrix()
            hits = model.collisions(q[finger_ids], center, rotation)
            snapshots[prefix or 'current'] = sorted(hits, key=lambda x: -x['depth_m'])
        entries.append(dict(step=event['step'], env=event['env'], age=event['age'],
                            force_N=float(np.linalg.norm(event['digit_forces'], axis=1).max()),
                            object_speed_m_s=float(np.linalg.norm(event['object'][7:10])),
                            intersections=snapshots))
    results.append(dict(run=path.parent.name, diagnostics=report['diagnostics'], events=entries))
(base / 'anomaly_geometry.json').write_text(json.dumps(results, indent=2))
for result in results:
    print(result['run'], result['diagnostics'])
    if result['events']:
        print('FIRST_EVENT', result['events'][0])
