
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT / 'outputs/nov20/ax50_brush'
OUT = P / 'continue'
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
from continue_opposed_trial import prepare, Q, MOUNT, R
import newton


def main():
    r = AX50WritingScene(side='left', wrist='locked', position=MOUNT, orientation=Q,
                         gravity=(0, 0, 0), iterations=60,
                         brush_xml=P/'tongji_scene_source.xml')
    r.restore_prepared_hand(np.load(prepare()))
    qbase = r.state.joint_q.numpy().copy()
    lower = r.model.joint_limit_lower.numpy()
    upper = r.model.joint_limit_upper.numpy()
    names = ['thumb_sweep', 'digit_1_flex_1', 'digit_1_flex_2',
             'digit_2_flex_1', 'digit_2_flex_2', 'digit_2_flex_3',
             'digit_3_flex_1', 'digit_3_flex_2', 'digit_3_flex_3']
    joint_q_start = r.model.joint_q_start.numpy()
    joint_qd_start = r.model.joint_qd_start.numpy()
    joint_parent = r.model.joint_parent.numpy()
    print('MODEL_JOINTS', json.dumps([{'joint_index': j, 'label': label,
             'q_start': int(joint_q_start[j]), 'qd_start': int(joint_qd_start[j]),
             'parent': int(joint_parent[j])}
            for j, label in enumerate(r.model.joint_label)]), flush=True)
    print('JOINTS', json.dumps({n: {'index': r.joint_names.index(n),
                                    'current': float(qbase[r.joint_names.index(n)]),
                                    'bounds': [float(lower[r.joint_names.index(n)]),
                                               float(upper[r.joint_names.index(n)])]} for n in names}), flush=True)
    pad_bones = np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
    shape_body = r.model.shape_body.numpy()
    shape_transform = r.model.shape_transform.numpy()
    shapes = {label: i for i, label in enumerate(r.model.shape_label)}
    tracked = [label for label in shapes if label.endswith(('skin_1_2', 'skin_3_3'))]

    def evaluate(overrides):
        q = qbase.copy()
        for name, value in overrides.items():
            q[r.joint_names.index(name)] = value
        r.state.joint_q.assign(q)
        newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
        body = r.state.body_q.numpy()
        particles = r.state.particle_q.numpy()
        result = {}
        for label in tracked:
            si = shapes[label]
            mesh = r.model.shape_source[si]
            points = Rotation.from_quat(shape_transform[si, 3:]).apply(np.asarray(mesh.vertices, dtype=float)) + shape_transform[si, :3]
            points = Rotation.from_quat(body[shape_body[si], 3:]).apply(points) + body[shape_body[si], :3]
            result[label.rsplit('/', 1)[-1]] = R.inv().apply(points-MOUNT)
        for bone in ['link_2_1', 'link_2_2', 'link_2_3']:
            points = particles[np.flatnonzero(pad_bones == bone) + r.pad_offset]
            result[bone] = R.inv().apply(points-MOUNT)
        return result

    directions = {}
    for name in names:
        index = r.joint_names.index(name)
        lo, hi = max(lower[index], -.2), min(upper[index], 1.5)
        for value in sorted(set([qbase[index], lo, (lo+hi)/2, hi])):
            points = evaluate({name: value})
            row = {'value': float(value)}
            for key, xyz in points.items():
                row[key] = {'center': xyz.mean(0).round(6).tolist(),
                            'extent': [xyz.min(0).round(6).tolist(), xyz.max(0).round(6).tolist()]}
            directions.setdefault(name, []).append(row)
    (OUT/'joint_directions.json').write_text(json.dumps(directions, indent=2))
    for name, rows in directions.items():
        print('MOTION', name, json.dumps([{'q': row['value'],
                                           'link_2_3': row['link_2_3']['center'],
                                           'skin_1_2': row['skin_1_2']['center'],
                                           'skin_3_3': row['skin_3_3']['center']} for row in rows]), flush=True)

    best = []
    for f1 in np.arange(.8, 1.51, .1):
        for f2 in [.7, 1., 1.3, 1.5]:
            for f3 in [0., .4, .7]:
                overrides = {'digit_3_flex_1': float(f1),
                             'digit_3_flex_2': f2, 'digit_3_flex_3': f3}
                points = evaluate(overrides)
                index_center = points['link_2_3'].mean(0)
                thumb_center = points['skin_1_2'].mean(0)
                middle_center = points['skin_3_3'].mean(0)
                center = np.array([.038, .044, middle_center[2]])
                distances = {}
                for key, xyz in points.items():
                    delta = xyz-center
                    radial = np.linalg.norm(delta[:, :2], axis=1)-.005
                    k = np.argmin(radial)
                    distances[key] = {'gap_mm': float(radial[k]*1000),
                                      'point': xyz[k].tolist()}
                score = abs(distances['skin_3_3']['gap_mm'])
                best.append({'joints': overrides, 'center': center.tolist(), 'score': score,
                                 'contacts_mm': distances})
    best.sort(key=lambda x: x['score'])
    (OUT/'closure_scan.json').write_text(json.dumps({'candidates': best}, indent=2))
    print('CANDIDATES', len(best), flush=True)
    for row in best[:15]:
        print('CANDIDATE', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
