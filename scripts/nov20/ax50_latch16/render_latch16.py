"""Render AX50 latch-16 dual-view video (left: full scene with AX50 skin, right: mechanism only) and curves."""

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
import numpy as np
import mujoco as mj
import trimesh

ROOT = Path((str(_EVOLUTION_ROOT)))
pkg = ROOT / 'outputs/ax50_visualskin_v3_0_0'
sys.path[:0] = [str(pkg), str(pkg / 'tools')]
import ax50_visualskin  # noqa: F401
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
import subprocess

OUT = ROOT / 'outputs/nov20/ax50_latch16'
mode = sys.argv[1] if len(sys.argv) > 1 else 'hand'
z = np.load(OUT / f'{mode}.npz')
poses = z['body_q']
times = z['time']
bolt_mm = z['bolt_mm']
lid_deg = z['lid_deg']
skin = VisualSkin('left', list(z['body_names']), device='cpu')
m = mj.MjModel.from_xml_path(str(OUT / 'latch_clean.xml'))
d = mj.MjData(m)
soft_tri = z['tri']
soft_tri = soft_tri[(soft_tri >= 0).all(axis=1) & (soft_tri < z['particles'].shape[1]).all(axis=1)]

# Joint qpos addresses in clean XML
q_bolt_adr = m.jnt_qposadr[m.joint('slide_bolt').id]
q_lid_adr = m.jnt_qposadr[m.joint('lid_hinge').id]

parts = []
for g in range(m.ngeom):
    name = m.geom(g).name
    if name == 'floor':
        continue
    typ = m.geom_type[g]
    sz = m.geom_size[g]
    if typ == mj.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=sz * 2)
        v, f = np.array(mesh.vertices), np.array(mesh.faces)
    elif typ == mj.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=sz[0], height=sz[1] * 2, count=[8, 12])
        v = np.array(mesh.vertices)
        v -= v.mean(0)
        f = np.array(mesh.faces)
    else:
        continue
    parts.append((g, v, f, np.array(m.geom_rgba[g, :3]) * 255.0, name))

def camera(aim, view, scale, w=600, h=448):
    zz = np.asarray(view, dtype=float)
    zz /= np.linalg.norm(zz)
    xx = np.cross([0, 0, 1], zz)
    xx /= np.linalg.norm(xx)
    b = np.array([xx, -np.cross(zz, xx), zz]).T
    return b, scale, np.array([w / 2.0, h / 2.0, 0.0]) - np.asarray(aim) @ b * scale

cam_left = camera([-0.155, 0.012, 0.095], [0.55, -0.75, 0.45], 2400)
cam_right = camera([-0.185, 0.018, 0.088], [0.65, 0.65, 0.45], 3200)

def render_frame(i):
    d.qpos[q_bolt_adr] = bolt_mm[i] / 1000.0
    d.qpos[q_lid_adr] = np.deg2rad(lid_deg[i])
    mj.mj_kinematics(m, d)
    obj_meshes = []
    for g, v, f, col, name in parts:
        obj_meshes.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], f, col, name))
    v_skin, f_skin = skin.update(poses[i])
    hand_meshes = [
        (v_skin, f_skin, [222, 174, 143], 'AX50 display skin'),
        (z['particles'][i], soft_tri, [235, 187, 156], 'AX50 soft'),
    ]
    left = panel(hand_meshes + obj_meshes, cam_left)
    right = panel(obj_meshes, cam_right)
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.hstack([left, right]))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 1200, 26], fill=(22, 30, 34))
    dr.text((10, 6), f'{mode} | AX50 | t={times[i]:.2f}s', fill='white')
    dr.text((610, 6), f'Mechanism view | bolt={bolt_mm[i]:.1f}mm | lid={lid_deg[i]:.1f}deg', fill='white')
    return np.asarray(im)

# Render at 25 fps (stride=4 from 100Hz, or stride=2 for 50fps -> 2x slow-mo at 25fps)
stride = 2
frames = [render_frame(i) for i in range(0, len(times), stride) if np.isfinite(poses[i]).all()]
h_img, w_img = frames[0].shape[:2]
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w_img}x{h_img}', '-r', '25', '-i', '-', '-pix_fmt', 'yuv420p', str(OUT / f'{mode}.mp4')], input=np.asarray(frames, dtype=np.uint8).tobytes(), check=True)
print(f'RENDERED {mode}.mp4 frames={len(frames)}', flush=True)

# Plot curves
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
trace = list(csv.DictReader((OUT / f'{mode}_trace.csv').open()))
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
axes[0].plot(times, bolt_mm, label='slide_bolt (mm)')
axes[0].axhline(8.0, color='r', linestyle='--', label='unlock 8 mm')
axes[0].set_ylabel('Bolt (mm)')
axes[0].legend()
axes[1].plot(times, lid_deg, label='lid_hinge (deg)')
axes[1].axhline(60.0, color='g', linestyle='--', label='open 60 deg')
axes[1].axhline(3.0, color='r', linestyle=':', label='locked tol 3 deg')
axes[1].set_ylabel('Lid (deg)')
axes[1].legend()
axes[2].plot(times, [float(r['index_bolt_N']) for r in trace], label='index-bolt N')
axes[2].plot(times, [float(r['index_lid_N']) for r in trace], label='index-lid N')
axes[2].plot(times, [float(r['ear_tongue_N']) for r in trace], label='ear-tongue N')
axes[2].set_ylabel('Normal force (N)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
plt.tight_layout()
plt.savefig(OUT / f'{mode}_curves.png', dpi=150)
print(f'SAVED {mode}_curves.png', flush=True)
