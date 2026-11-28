"""FK probe: choose hand mount orientation/position over the four keys."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from keys_runtime import *
import json
from scipy.spatial.transform import Rotation as Rot
out = ROOT / 'outputs/nov20/ax50_keys17'

# Q1: +X->-X, +Y->-Z, +Z->-Y  (xyzw)
Q1 = np.array([0.0, 0.7071067811865476, -0.7071067811865476, 0.0])
# Q2 = world-Z 180 deg then Q1 (scipy composes: q2 * q1)
Q2 = (Rot.from_euler('z', 180, degrees=True) * Rot.from_quat(Q1)).as_quat()

def report(tag, h):
    q = h.state.joint_q.numpy()
    bq = h.state.body_q.numpy()
    names = h.body_names_all
    print(f'=== {tag} keys q(mm) {np.round(q[h.key_q]*1000, 4)}', flush=True)
    tips = {}
    for n in ['link_palm', 'link_2_0', 'link_2_1', 'link_2_2', 'link_2_3',
              'link_3_0', 'link_3_1', 'link_3_2', 'link_3_3',
              'link_4_0', 'link_4_1', 'link_4_2', 'link_4_3',
              'link_5_0', 'link_5_1', 'link_5_2', 'link_5_3']:
        i = names.index(n)
        p = bq[i, :3]
        print(f'  {n:12s} {np.round(p, 4)}', flush=True)
        if n.endswith('_3'):
            tips[n] = p
    names_t = ['link_2_3', 'link_3_3', 'link_4_3', 'link_5_3']
    pts = np.array([tips[n] for n in names_t])
    print('  tip centroid', np.round(pts.mean(0), 4), 'lateral span', np.round(pts.max(0) - pts.min(0), 4), flush=True)
    pq = h.state.particle_q.numpy()
    soft = pq[:h.real_count]
    print('  soft z min/max', np.round(soft[:, 2].min(), 4), np.round(soft[:, 2].max(), 4), flush=True)
    print('  soft x range', np.round(soft[:, 0].min(), 4), np.round(soft[:, 0].max(), 4),
          'y range', np.round(soft[:, 1].min(), 4), np.round(soft[:, 1].max(), 4), flush=True)
    # nearest soft particle to each key top centre
    for k, c in enumerate(CENTERS):
        d = np.linalg.norm(soft - c, axis=1)
        j = int(d.argmin())
        print(f'  key{k+1} nearest soft {np.round(soft[j], 4)} dist {d[j]*1000:.2f}mm', flush=True)
    return pts

for tag, Q in [('Q1', Q1), ('Q2', Q2)]:
    h = KeysRuntime(side='right', wrist='locked', position=(0, 0, .3),
                    orientation=tuple(Q), gravity=(0, 0, -9.81))
    pts = report(tag, h)
    if tag == 'Q1':
        palm = h.state.body_q.numpy()[h.body_names_all.index('link_palm'), :3]
        off = pts.mean(0) - palm
        print('  fingertip centroid - palm', np.round(off, 4), flush=True)
        print('  target palm for tip centroid over keys', np.round(CENTERS.mean(0) - off, 4), flush=True)
    del h
