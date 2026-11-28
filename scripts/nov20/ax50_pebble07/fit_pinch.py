"""CPU thumb/index candidate using measured terminal geometry; no grasp certification."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path((str(_EVOLUTION_ROOT)))
OUT = ROOT / 'outputs/nov20/ax50_pebble07'
m = mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'))
d = mj.MjData(m)
ids = [next(i for i in range(m.nbody) if m.body(i).name.endswith(n)) for n in ['link_1_2', 'link_2_3']]
slots = np.array([0, 1, 2, 3, 16, 17, 18, 19, 20])
joint_names = [m.joint(int(i)).name.split('_Joints_')[-1] for i in slots]
# Mesh-informed distal cap points; cap normals are planning approximations, not measured contact normals.
points = [np.array([0, .0036, .0235]), np.array([0, .0031, .0195])]
normals=np.array([[0,.0036,.0025],[0,.0031,.00125]]);normals/=np.linalg.norm(normals,axis=1)[:,None]
lo = m.jnt_range[slots, 0] + .05
hi = m.jnt_range[slots, 1] - .05
seed = np.array([.2, .55, .55, .45, .45, .2, .3, .55, .45])
seed = np.clip(seed, lo, hi)

def fk(x):
    d.qpos[:] = 0
    d.qpos[slots] = x
    mj.mj_kinematics(m, d)
    p = np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ v for b, v in zip(ids, points)])
    n = np.array([d.xmat[b].reshape(3, 3) @ normal for b,normal in zip(ids,normals)])
    return p, n

def err(x):
    p, n = fk(x)
    delta = p[1] - p[0]
    length = np.linalg.norm(delta)
    axis = delta / length
    return np.r_[(length - .020) * 1000, (n[0] - axis) * 2, (n[1] + axis) * 2, (x - seed) * .1]

r = least_squares(err, seed, bounds=(lo, hi), max_nfev=500)
p, n = fk(r.x)
axis = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
approach = p.mean(0) - np.array([0, 0, .3])
approach -= axis * (axis @ approach)
approach /= np.linalg.norm(approach)
rot, _ = R.align_vectors(np.array([[0, 1, 0], [0, 0, -1]]), np.array([axis, approach]))
center = np.array([-.093, .003, .0345])
mount = center - rot.apply(p.mean(0) - [0, 0, .3])
q = np.zeros(21)
q[slots] = r.x
for k in [4,8,12]:q[k:k+4]=[.1,1.,1.2,1.]
margins = np.minimum(r.x - m.jnt_range[slots, 0], m.jnt_range[slots, 1] - r.x)
result = dict(scope='CPU FK candidate only; mesh-informed cap points and approximate normals, no collision or grasp validation',
              terminal_bodies=['link_1_2', 'link_2_3'], joint_names=joint_names,
              joint_values=r.x.tolist(), joint_margins_rad=margins.tolist(), min_joint_margin_rad=float(margins.min()),
              joint_pose=q.tolist(), position=mount.tolist(), orientation=rot.as_quat().tolist(),
              gap_mm=float(np.linalg.norm(p[1] - p[0]) * 1000),
              normal_alignment=[float(n[0] @ axis), float(-n[1] @ axis)],
              local_points=[v.tolist() for v in points], pad_normals_local=normals.tolist(),
              solver_success=bool(r.success))
(OUT/'pinch_candidate.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
