"""Render AX50 fruit-02 video (with breakable stem capsule) and force curves."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import csv
import subprocess
import sys
import numpy as np
import mujoco as mj
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = Path((str(_EVOLUTION_ROOT)))
pkg = ROOT / "outputs/ax50_visualskin_v3_0_0"
sys.path[:0] = [str(pkg), str(pkg / "tools")]
import ax50_visualskin  # noqa: F401
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel

OUT = ROOT / "outputs/nov20/ax50_fruit02"
tag = sys.argv[1] if len(sys.argv) > 1 else "pick_side_s78"
z = np.load(OUT / f"{tag}.npz")
poses = z["body_q"]
times = z["time"]
attached = z["attached"]
stem_N = z["stem_N"]
thumb_N = z["thumb_N"]
index_N = z["index_N"]
names = list(z["body_names"])
i_fr = names.index("fruit")
skin = VisualSkin("left", names, device="cpu")
m = mj.MjModel.from_xml_path(str(OUT / "fruit_clean.xml"))
soft_tri = z["tri"]
soft_tri = soft_tri[(soft_tri >= 0).all(axis=1) & (soft_tri < z["particles"].shape[1]).all(axis=1)]

# Build static branch mesh & fruit mesh
branch_mesh = trimesh.creation.capsule(radius=0.004, height=np.linalg.norm(np.array([-0.093, 0.003, 0.127]) - np.array([-0.195, 0.016, 0.135])), count=[8, 12])
p0, p1 = np.array([-0.195, 0.016, 0.135]), np.array([-0.093, 0.003, 0.127])
vec = p1 - p0
v_dir = vec / np.linalg.norm(vec)
rot_b = R.align_vectors([v_dir], [[0, 0, 1]])[0].as_matrix()
bv = np.array(branch_mesh.vertices) @ rot_b.T + (p0 + p1) / 2.0
bf = np.array(branch_mesh.faces)

# Fruit mesh from clean XML
f_id = m.geom("berry").dataid[0]
fv0 = m.mesh_vert[m.mesh_vertadr[f_id]:m.mesh_vertadr[f_id] + m.mesh_vertnum[f_id]].copy()
ff0 = m.mesh_face[m.mesh_faceadr[f_id]:m.mesh_faceadr[f_id] + m.mesh_facenum[f_id]].copy()

ANCHOR = np.array([-0.093, 0.003, 0.127])

def make_stem_mesh(end_pt):
    dv = end_pt - ANCHOR
    L = max(float(np.linalg.norm(dv)), 1e-4)
    cap = trimesh.creation.capsule(radius=0.0014, height=L, count=[6, 8])
    rmat = R.align_vectors([dv / L], [[0, 0, 1]])[0].as_matrix()
    return np.array(cap.vertices) @ rmat.T + (ANCHOR + end_pt) / 2.0, np.array(cap.faces)

def camera(aim, view, scale, w=600, h=448):
    zz = np.asarray(view, dtype=float)
    zz /= np.linalg.norm(zz)
    xx = np.cross([0, 0, 1], zz)
    xx /= np.linalg.norm(xx)
    b = np.array([xx, -np.cross(zz, xx), zz]).T
    return b, scale, np.array([w / 2.0, h / 2.0, 0.0]) - np.asarray(aim) @ b * scale

cam_front = camera([-0.090, 0.003, 0.075], [-0.75, -0.60, 0.22], 2700)
cam_palm = camera([-0.090, 0.003, 0.075], [-0.25, 0.95, 0.15], 2700)

frames = []
for i in range(0, len(times), 2):
    if not np.isfinite(poses[i]).all():
        continue
    fp = poses[i, i_fr, :3]
    fq = poses[i, i_fr, 3:]
    rmat = R.from_quat(fq).as_matrix()
    fv = fv0 @ rmat.T + fp
    end_pt = fp + rmat @ np.array([0.0, 0.0, 0.014]) if attached[i] else ANCHOR + np.array([0.0, 0.0, -0.025])
    sv, sf = make_stem_mesh(end_pt)
    v_skin, f_skin = skin.update(poses[i])
    meshes = [
        (v_skin, f_skin, [222, 174, 143], "AX50 skin"),
        (z["particles"][i], soft_tri, [235, 187, 156], "AX50 soft"),
        (bv, bf, [95, 54, 26], "branch"),
        (sv, sf, [110, 138, 38], "stem"),
        (fv, ff0, [230, 41, 12], "fruit"),
    ]
    left = np.asarray(panel(meshes, cam_front))
    right = np.asarray(panel(meshes, cam_palm))
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.hstack([left, right]))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 1200, 26], fill=(22, 30, 34))
    st_lbl = "ATTACHED" if attached[i] else "DETACHED"
    dr.text((10, 6), f"Front grasp view | 02 {tag} | t={times[i]:.2f}s | {st_lbl}", fill="white")
    dr.text((610, 6), f"Palm-side grasp view | stem={stem_N[i]:.2f}N | th={thumb_N[i]:.2f}N | ix={index_N[i]:.2f}N", fill="white")
    frames.append(np.asarray(im))

h_img, w_img = frames[0].shape[:2]
subprocess.run(
    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_img}x{h_img}", "-r", "25", "-i", "-", "-pix_fmt", "yuv420p", str(OUT / f"{tag}.mp4")],
    input=np.asarray(frames, dtype=np.uint8).tobytes(),
    check=True,
)
print(f"RENDERED {tag}.mp4 frames={len(frames)}", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(times, stem_N * attached, label="applied stem tension (N)")
ax[0].axhline(0.45, color="r", linestyle="--", label="break 0.45 N")
ax[0].set_ylabel("Stem (N)"); ax[0].legend()
ax[1].plot(times, thumb_N, label="thumb_distal (N)")
ax[1].plot(times, index_N, label="index_distal (N)")
ax[1].set_ylabel("Contact (N)"); ax[1].set_xlabel("Time (s)"); ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / f"{tag}_curves.png", dpi=150)
print(f"SAVED {tag}_curves.png", flush=True)
