"""Measure actual terminal-link rigid meshes and soft-pad vertices in body coordinates."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json
import numpy as np
import mujoco as mj

ROOT = Path((str(_EVOLUTION_ROOT)))
OUT = ROOT / 'outputs/nov20/ax50_pebble07'
m = mj.MjModel.from_binary_path(str(ROOT / 'outputs/nov20/ax50_keys17/left_ik.mjb'))
d = mj.MjData(m)
slots = np.array([0, 1, 2, 3, 16, 17, 18, 19, 20])
names = ['thumb_sweep', 'arch_1', 'thumb_axial', 'digit_1_flex_1', 'digit_1_flex_2',
         'digit_2_flex_1', 'digit_2_flex_2', 'digit_2_flex_3']
q = np.array([.05, .6, .8, .5, .6, .2, .3, .6, .5])
d.qpos[:] = 0
d.qpos[slots] = q
mj.mj_kinematics(m, d)
result = {'scope': 'CPU geometry measured from actual left_ik.mjb', 'joints': {}, 'bodies': {}}
for name, j, value in zip(names, slots, q):
    result['joints'][name] = {'slot': int(j), 'seed_rad': float(value),
                              'range_rad': m.jnt_range[j].tolist()}
for short in ['link_1_2', 'link_2_3']:
    body = next(i for i in range(m.nbody) if m.body(i).name.endswith(short))
    geoms = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != body:
            continue
        info = {'geom': g, 'name': mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g),
                'type': int(m.geom_type[g]), 'pos': m.geom_pos[g].tolist(),
                'quat_wxyz': m.geom_quat[g].tolist(), 'size': m.geom_size[g].tolist()}
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            k = int(m.geom_dataid[g])
            v = np.array(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]], float)
            info.update(vertex_count=len(v), min=v.min(0).tolist(), max=v.max(0).tolist(),
                        center=v.mean(0).tolist())
        geoms.append(info)
    mj.mj_forward(m, d)
    contacts = []
    for c in range(d.ncon):
        con = d.contact[c]
        if int(m.geom_bodyid[con.geom1]) == body or int(m.geom_bodyid[con.geom2]) == body:
            contacts.append(con.dist)
    result['bodies'][short] = {'body': int(body), 'body_pos': m.body_pos[body].tolist(),
                               'body_quat_wxyz': m.body_quat[body].tolist(), 'geoms': geoms}

pad_paths = list((ROOT / 'outputs/ax50_visualskin_v3_0_0').rglob('*pad*.npz'))
result['pad_asset_candidates'] = [str(x) for x in pad_paths]
(OUT / 'terminal_geometry.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
