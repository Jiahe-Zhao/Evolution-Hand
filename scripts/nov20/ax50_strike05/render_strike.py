
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import csv
import json
import sys

ROOT = Path((str(_EVOLUTION_ROOT)))
PKG = ROOT / 'outputs/ax50_visualskin_v3_0_0'
OUT = ROOT / 'outputs/nov20/ax50_strike05'
sys.path[:0] = [str(PKG), str(PKG / 'tools')]
import mujoco as mj
import numpy as np
from ax50_visualskin.visual import VisualSkin
from scipy.spatial.transform import Rotation
from smooth_render import panel
from render_core import encode, strip

W, H = 600, 448


def box(size):
    x, y, z = size
    v = np.array([[-x if i < 4 else x, -y if (i // 2) % 2 == 0 else y, -z if i % 2 == 0 else z] for i in range(8)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def ellipsoid(size):
    u, v = np.meshgrid(np.linspace(0., 2. * np.pi, 32, endpoint=False), np.linspace(-np.pi / 2., np.pi / 2., 17))
    points = np.column_stack([np.cos(v).ravel() * np.cos(u).ravel(), np.cos(v).ravel() * np.sin(u).ravel(), np.sin(v).ravel()]) * size
    faces = []
    for j in range(16):
        for i in range(32):
            a = j * 32 + i; b = j * 32 + (i + 1) % 32
            faces.extend([[a, b, b + 32], [a, b + 32, a + 32]])
    return points, np.array(faces)


def cylinder(size):
    r, h = size
    a = np.linspace(0., 2. * np.pi, 48, endpoint=False)
    p = np.vstack([np.column_stack([r * np.cos(a), r * np.sin(a), np.full(48, -h)]),
                   np.column_stack([r * np.cos(a), r * np.sin(a), np.full(48, h)])])
    f = [[i, (i + 1) % 48, (i + 1) % 48 + 48] for i in range(48)]
    f += [[i, (i + 1) % 48 + 48, i + 48] for i in range(48)]
    f += [[0, i + 1, i] for i in range(1, 47)]
    f += [[48, 48 + i, 48 + i + 1] for i in range(1, 47)]
    return p, np.array(f)


def load_scene():
    m = mj.MjModel.from_xml_path(str(OUT / 'scene.xml'))
    geoms = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == 0 or m.geom_type[i] == mj.mjtGeom.mjGEOM_PLANE: continue
        typ = m.geom_type[i]
        if typ == mj.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[i]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        elif typ == mj.mjtGeom.mjGEOM_ELLIPSOID: v, f = ellipsoid(m.geom_size[i])
        elif typ == mj.mjtGeom.mjGEOM_CYLINDER: v, f = cylinder(m.geom_size[i])
        else: continue
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or ''
        if name.startswith(('stone_band', 'wood_grip')): continue
        body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ''
        color = tuple(int(255 * c) for c in m.geom_rgba[i, :3])
        if color == (0, 0, 0): color = (124, 121, 111) if name.startswith('stone') else (159, 113, 61)
        geoms.append(dict(body=body, name=name, v=np.array(v), f=np.array(f), color=color,
                          pos=m.geom_pos[i], quat=m.geom_quat[i][[1, 2, 3, 0]]))
    return geoms


def camera(points, view):
    zz = np.array(view, float); zz /= np.linalg.norm(zz)
    xx = np.cross([0, 0, 1], zz); xx /= np.linalg.norm(xx)
    basis = np.array([xx, -np.cross(zz, xx), zz]).T
    points = np.vstack(points)
    center = (points.min(0) + points.max(0)) / 2.
    delta = points - center
    sx = max(np.max(np.abs(delta @ basis[:, 0])), 1e-5)
    sy = max(np.max(np.abs(delta @ basis[:, 1])), 1e-5)
    scale = min(.88 * W / 2 / sx, .88 * H / 2 / sy)
    return basis, scale, np.array([W / 2, H / 2, 0.]) - center @ basis * scale


def render(tag):
    z = np.load(OUT / f'{tag}.npz')
    time = z['time'].ravel()
    body_q = z['body_q']
    names = [str(n).rsplit('/', 1)[-1] for n in z['body_names']]
    geoms = load_scene()
    skin = VisualSkin('left', names, device='cpu')
    tri = z['tri']; particles = z['particles']
    valid_tri = tri[(tri >= 0).all(1) & (tri < particles.shape[1]).all(1)]
    rows = list(csv.DictReader((OUT / f'{tag}_trace.csv').open()))
    result = json.loads((OUT / f'{tag}_result.json').read_text())

    def world(i):
        pb, pq = body_q[i, :, :3], body_q[i, :, 3:]
        rb = Rotation.from_quat(pq).as_matrix()
        parts = []
        for g in geoms:
            if g['body'] in names:
                j = names.index(g['body'])
                m = rb[j] @ Rotation.from_quat(g['quat']).as_matrix()
                p = pb[j] + rb[j] @ g['pos']
            else:
                m = Rotation.from_quat(g['quat']).as_matrix(); p = g['pos']
            parts.append((g['v'] @ m.T + p, g['f'], g['color'], g['name']))
        return parts
    samples = list(range(0, len(time), max(1, len(time) // 10)))
    points = []
    for i in samples:
        points.extend(g[0] for g in world(i))
        points.append(skin.update(body_q[i])[0])
        points.append(particles[i])
    cam = camera(points, [-.65, -1., .42])
    skin_cam = camera(points, [-.50, -.65, .85])
    frames = np.unique(np.searchsorted(time, np.arange(time[0], time[-1] + 1.e-9, 1. / 30)).clip(0, len(time) - 1))

    def frame(i):
        parts = world(i)
        v, f = skin.update(body_q[i])
        parts.extend([(v, f, (222, 174, 143), 'AX50 skin'), (particles[i], valid_tri, (235, 187, 156), 'soft')])
        r = rows[min(int(round((time[i] - float(rows[0]['time_s'])) / float(rows[0]['dt']))), len(rows) - 1)]
        text = [f"t {time[i]:.3f} s   phase {r['phase']}",
                f"stone-target {float(r['stone_target_N']):.3f} N   hand-target {float(r['hand_target_N']):.3f} N",
                f"hold {float(r['best_hold_s']):.3f} s   penetration {float(r['max_penetration_m']) * 1000:.3f} mm"]
        passed = result.get('grasp_acceptance', result.get('strike_acceptance', False))
        return strip([panel(parts, cam), panel(parts, skin_cam)], ['whole hand and original hammer', 'target and contact'],
                     [text, text], f"AX50 strike05 recorded Newton replay | {'PASS' if passed else 'NOT PASSED'}")
    encode(OUT / f'{tag}.mp4', 15, (frame(i) for i in frames))

    import subprocess
    decoded = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                              '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(OUT / f'{tag}.mp4')],
                             text=True, capture_output=True)
    if decoded.returncode:
        raise RuntimeError(decoded.stderr)
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(OUT / f'{tag}.mp4'), '-f', 'null', '-'],
                            text=True, capture_output=True)
    decoded_frames = int(json.loads(decoded.stdout)['streams'][0]['nb_read_frames'])
    if decode.returncode or decoded_frames != len(frames):
        raise RuntimeError(f'decode failed: rc={decode.returncode} {decode.stderr} frames={decoded_frames} expected={len(frames)}')
    decode_info = dict(ffmpeg_decode_rc=decode.returncode, decoded_frames=decoded_frames, expected_frames=len(frames))
    print(json.dumps(dict(tag=tag, **decode_info)), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([float(r['time_s']) for r in rows])
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name in ['stone_target_N', 'wood_target_N', 'hand_target_N']:
        ax[0].plot(t, [float(r[name]) for r in rows], label=name)
    ax[0].axhline(5., color='r', ls='--', label='stone threshold 5 N'); ax[0].legend(); ax[0].set_ylabel('normal force (N)')
    for name in ['thumb_N', 'index_N', 'middle_N', 'ring_N', 'little_N', 'external_support_N']:
        ax[1].plot(t, [float(r[name]) for r in rows], label=name)
    ax[1].legend(); ax[1].set_ylabel('hand / support force (N)')
    ax[2].plot(t, [float(r['max_penetration_m']) * 1000. for r in rows], label='max contact penetration')
    ax[2].plot(t, [float(r['best_hold_s']) for r in rows], label='best continuous hold')
    ax[2].legend(); ax[2].set_ylabel('mm / s'); ax[2].set_xlabel('simulation time (s)')
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / f'{tag}_force_curves.png', dpi=160); plt.close(fig)
    info = dict(tag=tag, frames=len(frames), fps=15, slow_motion='2x: measured 30 Hz poses encoded at 15 fps',
                decode_verification=decode_info,
                scope='MuJoCo reads unmodified asset geometry; recorded Newton poses/particles only; no invented contacts or physics',
                result=result, files=dict(npz=str(OUT / f'{tag}.npz'), mp4=str(OUT / f'{tag}.mp4'), curves=str(OUT / f'{tag}_force_curves.png')))
    (OUT / f'{tag}_render.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(dict(tag=tag, frames=len(frames), passed=info['result']), indent=2))


if __name__ == '__main__':
    render(sys.argv[1])
