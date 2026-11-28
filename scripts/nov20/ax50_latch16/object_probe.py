"""Object-only Newton MuJoCo-Warp probe for latch16 (locked vs clear) + AX50 smoke."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json
import numpy as np
from latch_runtime import *

OUT_DIR.mkdir(parents=True, exist_ok=True)
clean_summary = clean_latch_xml(SRC_XML, CLEAN_XML)

def run_probe(mode):
    b = newton.ModelBuilder(gravity=(0, 0, -9.81))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_joint_cfg.target_ke = 0.0
    b.default_joint_cfg.target_kd = 0.0
    b.add_mjcf(str(CLEAN_XML), collapse_fixed_joints=False)
    for j in range(len(b.joint_target_ke)):
        b.joint_target_ke[j] = 0.0
        b.joint_target_kd[j] = 0.0
        b.joint_target_mode[j] = newton.JointTargetMode.NONE
    names = [str(n).rsplit('/', 1)[-1] for n in b.joint_label]
    slide_j = names.index('slide_bolt')
    lid_j = names.index('lid_hinge')
    if mode == 'clear':
        b.joint_q[b.joint_q_start[slide_j]] = 0.018
    m = b.finalize(device='cuda:0')
    sq = int(m.joint_q_start.numpy()[slide_j])
    lq = int(m.joint_q_start.numpy()[lid_j])
    ld = int(m.joint_qd_start.numpy()[lid_j])
    s = m.state()
    nxt = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    ctrl = m.control()
    solver = newton.solvers.SolverMuJoCo(
        m, use_mujoco_contacts=False, integrator='implicitfast', nconmax=512, njmax=8192
    )
    pipe = newton.CollisionPipeline(m, broad_phase='sap', rigid_contact_max=512, soft_contact_max=1)
    c = pipe.contacts()
    dt = 1.0 / 240.0
    rows = []
    for t in range(240):
        f = np.zeros(m.joint_dof_count, np.float32)
        f[ld] = 0.025 * min(1.0, t / 30.0)
        ctrl.joint_f.assign(f)
        s.clear_forces()
        pipe.collide(s, c)
        solver.step(s, nxt, ctrl, c, dt)
        s, nxt = nxt, s
        q = s.joint_q.numpy()
        rows.append([t * dt, float(q[sq] * 1000.0), float(np.rad2deg(q[lq]))])
    a = np.array(rows)
    np.savetxt(OUT_DIR / f'probe_{mode}.csv', a, delimiter=',', header='time_s,bolt_mm,lid_deg', comments='')
    res = {
        'mode': mode,
        'max_bolt_mm': float(a[:, 1].max()),
        'max_lid_deg': float(a[:, 2].max()),
        'end_bolt_mm': float(a[-1, 1]),
        'end_lid_deg': float(a[-1, 2]),
        'finite': bool(np.isfinite(a).all()),
    }
    print('PROBE', res, flush=True)
    return res

r_locked = run_probe('locked')
r_clear = run_probe('clear')
(OUT_DIR / 'object_probe.json').write_text(json.dumps({'locked': r_locked, 'clear': r_clear, 'clean_summary': clean_summary}, indent=2))
