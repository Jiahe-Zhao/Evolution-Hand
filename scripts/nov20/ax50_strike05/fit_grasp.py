
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import hashlib
import json
import mujoco as mj
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path((str(_EVOLUTION_ROOT)))
OUT = ROOT / 'outputs/nov20/ax50_strike05'
HAND_PATH = ROOT / 'outputs/nov20/ax50_keys17/left_ik.mjb'
HAMMER_PATH = ROOT / 'outputs/nov20/hafted05/asset.xml'
CONVENTION = ROOT / 'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json'
NAMES = [n for i in range(2,6) for n in [f'arch_{i}', *[f'digit_{i}_flex_{j}' for j in range(1,4)]]]
model = mj.MjModel.from_binary_path(str(HAND_PATH))
data = mj.MjData(model)
signs = json.loads(CONVENTION.read_text())['canonical_to_native']
assert (model.nq, model.nv, model.ngeom, model.nbody) == (26, 26, 251, 36)
for i in range(2, 6):
    for link in range(4):
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'_ax50_left_Geometry_link_{i}_{link}') >= 0
for i in range(3):
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'_ax50_left_Geometry_link_1_{i}') >= 0


def jid(name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'_ax50_left_Joints_{name}')


def fk(q, fingers=None, thumb=None):
    data.qpos[:] = 0
    for name,value in zip(NAMES,np.asarray(fingers).ravel()):
        data.qpos[model.jnt_qposadr[jid(name)]] = value * signs[name]
    for name, value in zip(['thumb_sweep', 'arch_1', 'thumb_axial', 'digit_1_flex_1', 'digit_1_flex_2'], thumb):
        data.qpos[model.jnt_qposadr[jid(name)]] = value * signs[name]
    mj.mj_kinematics(model, data)
    positions = data.xpos.copy() - [0, 0, .3]
    rotations = data.xquat.copy()[:, [1, 2, 3, 0]]
    return positions, rotations


def body(name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, '_ax50_left_Geometry_' + name)


def distance_model(fingers, thumb):
    positions, rotations = fk(None, fingers, thumb)
    spec = mj.MjSpec()
    spec.compiler.degree = False
    spec.option.disableflags = 1
    for mesh_id in range(model.nmesh):
        adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        fadr, fnum = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        mesh = spec.add_mesh()
        mesh.name = f'ax50_mesh_{mesh_id}'
        mesh.uservert = model.mesh_vert[adr:adr+num].astype(float).ravel()
        mesh.userface = model.mesh_face[fadr:fadr+fnum].astype(int).ravel()
    for geom_id in range(246):
        gid = model.geom_bodyid[geom_id]
        b = spec.worldbody.add_body(
            name=f'ax50_body_{geom_id}', pos=positions[gid], quat=rotations[gid][[3, 0, 1, 2]])
        geom = b.add_geom(name=f'ax50_geom_{geom_id}', type=int(model.geom_type[geom_id]),
                          pos=model.geom_pos[geom_id], quat=model.geom_quat[geom_id],
                          size=model.geom_size[geom_id], density=1.)
        if model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH:
            geom.meshname = f'ax50_mesh_{model.geom_dataid[geom_id]}'
    hammer = mj.MjSpec.from_file(str(HAMMER_PATH))
    hammer.body('hafted_hammer').pos = [0,0,0]
    frame = spec.worldbody.add_frame(pos=HAMMER_IN_HAND - np.array([0,0,-.085]), quat=[1,0,0,0])
    spec.attach(hammer, prefix='hammer_', frame=frame)
    static_model = spec.compile()
    static_data = mj.MjData(static_model)
    mj.mj_forward(static_model, static_data)
    return static_model, static_data


def audit(fingers, thumb, full=True):
    sm, sd = distance_model(fingers, thumb)
    hammer_ids = [i for i in range(sm.ngeom) if mj.mj_id2name(sm, mj.mjtObj.mjOBJ_GEOM, i).startswith('hammer_') and (sm.geom_contype[i] or sm.geom_conaffinity[i])]
    assert len(hammer_ids) == 20
    hand_ids = [i for i in range(246) if sm.geom_bodyid[i] != 0]
    records = []
    for a in hand_ids:
        body_id = model.geom_bodyid[a]
        link = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id).removeprefix('_ax50_left_Geometry_')
        for b in hammer_ids:
            if not full and not mj.mj_id2name(sm, mj.mjtObj.mjOBJ_GEOM, b).startswith('hammer_wood_'):
                continue
            fromto = np.zeros(6)
            dist = mj.mj_geomDistance(sm, sd, a, b, .12, fromto)
            records.append(dict(hand_geom=link, hammer_geom=mj.mj_id2name(sm, mj.mjtObj.mjOBJ_GEOM, b).removeprefix('hammer_'), distance_m=float(dist)))
    records.sort(key=lambda r: r['distance_m'])
    result = dict(scope='independent static CPU mesh distance audit, not Newton acceptance',
                  hand_model=str(HAND_PATH.relative_to(ROOT)),
                  hand_model_sha256=hashlib.sha256(HAND_PATH.read_bytes()).hexdigest(),
                  hammer_asset=str(HAMMER_PATH.relative_to(ROOT)),
                  hammer_asset_sha256=hashlib.sha256(HAMMER_PATH.read_bytes()).hexdigest(),
                  hand_collision_geoms=len(hand_ids), hammer_collision_geoms=len(hammer_ids),
                  min_signed_distance_m=records[0]['distance_m'], guard_limit_m=.0005,
                  initial_rigid_penetration_guard_would_pass=bool(records[0]['distance_m'] > -.0005),
                  worst_pairs=records[:100])
    (OUT / 'collision_audit.json').write_text(json.dumps(result, indent=2))
    return result


HAMMER_IN_HAND = np.array([.070, .015, .01])
Z = [-.006, .006, .017, .026]
fingers = []
for d, z in enumerate(Z, 2):
    def objective(q):
        p, _ = fk(None, np.asarray(fingers + [q] + [[0, 0, 0, 0]] * (4-len(fingers)-1)), [0] * 5)
        pip = p[body(f'link_{d}_2')]
        tip = p[body(f'link_{d}_3')]
        return np.r_[(pip - [.080, -.002, z]) / .012,
                     (tip - [.065, .025, z]) / .012, .1 * (q[0] - .12)]
    fit = least_squares(objective, [.12, .75, .8, .6], bounds=([-.1599, 0, 0, 0], [.7199, 1.4999, 1.4999, 1.4999]), max_nfev=500)
    fingers.append(fit.x.tolist())
    print('FINGER', d, fit.cost, fit.x, flush=True)


def thumb_objective(q):
    p, _ = fk(None, fingers, q)
    pip, tip = p[body('link_1_1')], p[body('link_1_2')]
    return np.r_[(pip - [.055,.020,-.015])/.018, (tip - [.070,.025,-.015])/.018]


thumb = least_squares(thumb_objective, [.25, .25, .3, .7, .65], bounds=([-.6899, -.1599, -1.4999, 0, 0], [.9499, .7199, 1.4999, 1.4999, 1.4999]), max_nfev=1000).x.tolist()
positions, rotations = fk(None, fingers, thumb)
plan = dict(scope='CPU candidate for original unchanged hammer and actual AX50; requires independent collision audit and actual Newton acceptance',
            model_identity=dict(model=str(HAND_PATH.relative_to(ROOT)), sha256=hashlib.sha256(HAND_PATH.read_bytes()).hexdigest(), nq=model.nq, nv=model.nv, ngeom=model.ngeom, nbody=model.nbody,
                                canonical_to_native={name: signs[name] for name in NAMES + ['thumb_sweep', 'arch_1', 'thumb_axial', 'digit_1_flex_1', 'digit_1_flex_2']}),
            fingers=fingers, thumb=thumb, hammer_in_hand=HAMMER_IN_HAND.tolist(), hand_axis='+y is hammer local +z through Rx(-90 degrees)',
            fk={f'link_{d}_{link}': positions[body(f'link_{d}_{link}')].tolist() for d in range(1, 6) for link in range(4) if body(f'link_{d}_{link}') >= 0})
(OUT / 'grasp_plan.json').write_text(json.dumps(plan, indent=2))
print('PLAN', json.dumps(dict(fingers=fingers, thumb=thumb, hammer_in_hand=HAMMER_IN_HAND.tolist())), flush=True)
result = audit(fingers, thumb)
print('AUDIT', json.dumps({k: v for k, v in result.items() if k != 'worst_pairs'}), flush=True)
def preview():
    import matplotlib
    matplotlib.use('Agg')
    sm, sd = distance_model(fingers, thumb)
    renderer = mj.Renderer(sm, height=640, width=800)
    renderer.update_scene(sd, camera=mj.MjvCamera())
    camera = renderer.scene.camera[0]
    camera.lookat[:] = [.065, .015, .005]
    camera.distance = .26
    camera.azimuth = 135
    camera.elevation = -25
    import matplotlib.pyplot as plt
    image = renderer.render()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title('Actual AX50 left_ik.mjb and original hammer | static geometry only')
    fig.tight_layout()
    fig.savefig(OUT / 'static_preview.png', dpi=150)
    plt.close(fig)
    renderer.close()
    print('PREVIEW', str(OUT / 'static_preview.png'), image.shape, flush=True)


# preview disabled pending validated camera
print('WORST', json.dumps(result['worst_pairs'][:10]), flush=True)
