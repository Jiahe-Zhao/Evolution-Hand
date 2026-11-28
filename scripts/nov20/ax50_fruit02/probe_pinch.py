
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

# Mount left hand: fingers toward -X, palm toward -Z (or tilted) to pinch fruit cheeks along Y
# Check thumb_distal (link_1_2) and index_distal (link_2_3) positions for opposed pinch
R_MOUNT = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 0.0, -1.0,  0.0],
], dtype=float)
QUAT = R.from_matrix(R_MOUNT).as_quat().tolist()
h = FruitRuntime(side="left", wrist="six_axis", position=[0.05, 0.0, 0.12], orientation=QUAT, gravity=(0, 0, -9.81))
i_th = h.body_names.index("link_1_2")
i_ix = h.body_names.index("link_2_3")
bq = h.state.body_q.numpy()
print("NEUTRAL thumb:", bq[i_th, :3].round(4), "index:", bq[i_ix, :3].round(4), "fruit:", bq[h.fruit_body, :3].round(4))

# Test a simple opposed pinch pose in air
t_pinch = np.zeros(len(h.action_names))
# index: flex_1, flex_2, flex_3
t_pinch[7] = 0.55; t_pinch[8] = 0.55; t_pinch[9] = 0.35
# thumb: thumb_sweep(22), arch_1(23), thumb_axial(24), flex_1(25), flex_2(26)
t_pinch[22] = 0.65; t_pinch[23] = 0.45; t_pinch[24] = 0.25; t_pinch[25] = 0.35; t_pinch[26] = 0.25
for step in range(40):
    u = min(1.0, (step + 1) / 30.0)
    h.set_targets(t_pinch * u)
    h.advance(0.01, dt=0.000125)
bq = h.state.body_q.numpy()
st = h.stem_state.numpy()
th_p, ix_p = bq[i_th, :3], bq[i_ix, :3]
print("PINCHED thumb:", th_p.round(4), "index:", ix_p.round(4), "dist_mm:", round(float(np.linalg.norm(th_p - ix_p) * 1000), 2), "mid:", ((th_p + ix_p) / 2).round(4))
print("FRUIT hanging:", bq[h.fruit_body, :3].round(4), "stem_state:", st.round(4))
