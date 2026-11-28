
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from latch_runtime import *

# Left hand: test mounting palm down (-Z), fingers pointing -X
for label, rot in [
    ('left_id', [0, 0, 0, 1]),
    ('left_rot', R.from_euler('xy', [90, -90], degrees=True).as_quat().tolist()),
]:
    h = LatchRuntime(side='left', wrist='six_axis', position=[0.0, 0.0, 0.20], orientation=rot, gravity=(0, 0, -9.81))
    bq = h.state.body_q.numpy()
    if label == 'left_id':
        print('body_names:', h.body_names)
        print('joint_names:', [str(x).rsplit('/', 1)[-1] for x in h.model.joint_label])
    idx_palm = h.body_names.index('link_palm')
    idx_tip = h.body_names.index('link_2_3')
    idx_mid = h.body_names.index('link_3_3')
    print(label, 'palm', bq[idx_palm, :3].round(4), 'index_tip', bq[idx_tip, :3].round(4), 'mid_tip', bq[idx_mid, :3].round(4))
    if label == 'left_rot':
        h.advance(0.05, dt=0.00025)
        q = h.state.joint_q.numpy()
        pv = h.state.particle_qd.numpy()
        print('smoke finite:', bool(np.isfinite(q).all() and np.isfinite(pv).all()), 'max_soft_v:', float(np.nanmax(np.linalg.norm(pv, axis=1))))
