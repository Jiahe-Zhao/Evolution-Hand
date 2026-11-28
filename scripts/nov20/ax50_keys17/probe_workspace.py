"""FK workspace: lowest point of each distal finger mesh vs joint angles."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from keys_runtime import *
from scipy.spatial.transform import Rotation as Rot

Q1 = (0.0, 0.7071067811865476, -0.7071067811865476, 0.0)
h = KeysRuntime(side='right', wrist='locked', position=(0, 0, .3),
                orientation=Q1, gravity=(0, 0, -9.81))
m = h.model
names = h.body_names_all
base_q = h.state.joint_q.numpy().copy()
act = {n: i for i, n in enumerate(h.action_names)}
shape_body = m.shape_body.numpy()
shape_tf = m.shape_transform.numpy()
srcs = m.shape_source

def fk(targets):
    q = base_q.copy()
    for n, v in targets.items():
        q[h.action_indices[act[n]]] = v * h.signs[act[n]]
    h.state.joint_q.assign(q)
    h.state.joint_qd.zero_()
    newton.eval_fk(m, h.state.joint_q, h.state.joint_qd, h.state)
    return h.state.body_q.numpy()

def lowest(body_name, bq):
    b = names.index(body_name)
    pts = []
    for i in range(len(shape_body)):
        if shape_body[i] != b or srcs[i] is None:
            continue
        v = np.asarray(srcs[i].vertices, dtype=float)
        tf = shape_tf[i]
        r = Rot.from_quat(tf[3:])
        v = r.apply(v) + tf[:3]
        r2 = Rot.from_quat(bq[b, 3:])
        pts.append(r2.apply(v) + bq[b, :3])
    if not pts:
        return None
    p = np.concatenate(pts)
    j = int(p[:, 2].argmin())
    return p[j], p[:, 2].min(), p[:, 2].max()

print('ref lowest points:', flush=True)
bq = fk({})
for d in [2, 3, 4, 5]:
    print(' digit', d, np.round(lowest(f'link_{d}_3', bq)[0], 4), flush=True)

for d in [2, 3, 4, 5]:
    print(f'--- digit {d} ---', flush=True)
    for f1 in [0.0, 0.5, 1.0, 1.5]:
        for arch in [0.0, 0.4, 0.8]:
            tg = {f'digit_{d}_flex_1': f1, f'arch_{d}': arch}
            p = lowest(f'link_{d}_3', fk(tg))[0]
            print(f'  f1={f1:.1f} arch={arch:.1f} low {np.round(p,4)}', flush=True)
# pinky distal z-axis joints
for f2 in [0.0, 0.8, 1.5]:
    for f3 in [0.0, 0.8, 1.5]:
        p = lowest('link_5_3', fk({'digit_5_flex_2': f2, 'digit_5_flex_3': f3}))[0]
        print(f'  pinky f2={f2:.1f} f3={f3:.1f} low {np.round(p,4)}', flush=True)
