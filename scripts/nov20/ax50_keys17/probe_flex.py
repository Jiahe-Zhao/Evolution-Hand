"""FK scan: how finger joint angles move each fingertip (no physics)."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from keys_runtime import *
import itertools

Q1 = (0.0, 0.7071067811865476, -0.7071067811865476, 0.0)
h = KeysRuntime(side='right', wrist='locked', position=(0, 0, .3),
                orientation=Q1, gravity=(0, 0, -9.81))
names = h.body_names_all
tips = ['link_2_3', 'link_3_3', 'link_4_3', 'link_5_3']
base_q = h.state.joint_q.numpy().copy()
act = {n: i for i, n in enumerate(h.action_names)}

def fk(targets):
    q = base_q.copy()
    for n, v in targets.items():
        q[h.action_indices[act[n]]] = v * h.signs[act[n]]
    h.state.joint_q.assign(q)
    h.state.joint_qd.zero_()
    newton.eval_fk(h.model, h.state.joint_q, h.state.joint_qd, h.state)
    bq = h.state.body_q.numpy()
    return {t: bq[names.index(t), :3] for t in tips}

ref = fk({})
print('ref tips', {k: np.round(v, 4).tolist() for k, v in ref.items()}, flush=True)
for digit in [2, 3, 4, 5]:
    t = f'link_{digit}_3'
    for kind in ['arch', 'flex1', 'flex2', 'flex3', 'all']:
        for val in ([0.4, 0.8] if kind != 'all' else [0.3, 0.6, 0.9]):
            if kind == 'arch':
                tg = {f'arch_{digit}': val}
            elif kind == 'all':
                tg = {f'digit_{digit}_flex_{j}': val for j in (1, 2, 3)}
            else:
                tg = {f'digit_{digit}_flex_{kind[-1]}': val}
            p = fk(tg)[t]
            print(f'digit{digit} {kind}={val:.1f} tip {np.round(p,4)} delta {np.round((p-ref[t])*1000,2)} mm', flush=True)
# thumb geometry
bq = h.state.body_q.numpy()
for n in ['thumb_cmc_base', 'thumb_cmc_rollbase', 'link_1_0', 'link_1_1', 'link_1_2', 'thenar', 'link_palm']:
    print('THUMB', n, np.round(bq[names.index(n), :3], 4), flush=True)
