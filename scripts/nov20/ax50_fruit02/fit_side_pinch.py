
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np
from scipy.spatial.transform import Rotation as R
from fruit_runtime import *

# Side-pinch mount: local +X -> world -X (fingers toward fruit), local +Y -> world +Y (palm faces +Y), local +Z -> world -Z
R_MOUNT = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=float)
QUAT = R.from_matrix(R_MOUNT).as_quat().tolist()
h = FruitRuntime(side="left", wrist="six_axis", position=[0.05, 0.0, 0.08], orientation=QUAT, gravity=(0, 0, -9.81))
i_th = h.body_names.index("link_1_2")
i_ix = h.body_names.index("link_2_3")
n_dof = len(h.action_names)

# Test open vs closed natural opposed pinch (thumb & index flex toward each other in palm-normal plane)
pose_open = np.zeros(n_dof)
for b in (10, 14, 18):
    pose_open[b + 1] = 0.45; pose_open[b + 2] = 0.45; pose_open[b + 3] = 0.25
pose_open[7] = 0.45; pose_open[8] = 0.45; pose_open[9] = 0.25
pose_open[22] = 0.55; pose_open[23] = 0.45; pose_open[24] = 0.30; pose_open[25] = 0.25; pose_open[26] = 0.20

pose_close = pose_open.copy()
pose_close[7] = 0.95; pose_close[8] = 0.90; pose_close[9] = 0.50
pose_close[22] = 0.85; pose_close[23] = 0.70; pose_close[24] = 0.45; pose_close[25] = 0.65; pose_close[26] = 0.50

for step in range(60):
    if step < 30:
        t = pose_open * min(1.0, (step + 1) / 25.0)
    else:
        u = min(1.0, (step - 29) / 25.0)
        t = pose_open + (pose_close - pose_open) * u
    h.set_targets(np.clip(t, h.action_lower, h.action_upper))
    h.advance(0.01, dt=0.000125)
    if step in (29, 59):
        bq = h.state.body_q.numpy()
        th, ix = bq[i_th, :3], bq[i_ix, :3]
        lbl = "SIDE_OPEN" if step == 29 else "SIDE_CLOSE"
        print(lbl, "th=", th.round(4), "ix=", ix.round(4), "mid=", ((th+ix)/2).round(4), "diff(ix-th)=", (ix-th).round(4), "dist_mm=", round(float(np.linalg.norm(ix-th)*1000), 1))
