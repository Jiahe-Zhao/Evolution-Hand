#!/usr/bin/env python3
"""Headless Isaac Sim import, ROM sweep, and RGB recording for fossil-hand URDFs."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np
import carb
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


def urdf_limits(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        limit = joint.find("limit")
        if joint.attrib.get("type") == "revolute" and limit is not None:
            result[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return result


def import_hand(path: Path) -> tuple[str, Articulation]:
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    config.merge_fixed_joints = True
    config.convex_decomp = False
    config.import_inertia_tensor = True
    config.fix_base = True
    config.distance_scale = 1.0
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(path), import_config=config, get_articulation_root=True
    )
    if not status:
        raise RuntimeError(f"URDF import failed: {path}")
    art = Articulation(prim_paths_expr=prim_path, name="fossil_hand")
    return prim_path, art


def setup_scene() -> None:
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    light.CreateIntensityAttr(2200.0)
    light.CreateAngleAttr(0.45)
    stage.DefinePrim("/World", "Xform")
    backdrop = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Backdrop"))
    backdrop.CreateSizeAttr(1.0)
    backdrop.AddTranslateOp().Set(Gf.Vec3d(0.22, 0.0, 0.07))
    backdrop.AddScaleOp().Set(Gf.Vec3f(0.01, 0.8, 0.8))
    backdrop.CreateDisplayColorAttr([Gf.Vec3f(0.035, 0.055, 0.065)])
    set_camera_view(eye=[-0.38, -0.29, 0.24], target=[0.025, 0.0, 0.07], camera_prim_path="/OmniverseKit_Persp")


def color_imported_hand() -> None:
    stage = omni.usd.get_context().get_stage()
    palette = {
        "chassis": (0.10, 0.16, 0.19),
        "shell": (0.35, 0.44, 0.47),
        "rail": (0.72, 0.78, 0.76),
        "hinge": (0.88, 0.43, 0.16),
        "pad": (0.08, 0.61, 0.53),
    }
    materials = {}
    for name, rgb in palette.items():
        material = UsdShade.Material.Define(stage, Sdf.Path(f"/Looks/Mechanical_{name}"))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(f"/Looks/Mechanical_{name}/Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.65 if name != "pad" else 0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.36 if name != "pad" else 0.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = str(prim.GetPath())
        if "/Looks/" in path or "Backdrop" in path or "collision" in path.lower():
            continue
        match = re.search(r"link_[1-5]_(\d+)", path)
        if "link_palm" in path or "link_0_0" in path:
            color = "chassis"
        elif "spread" in path:
            color = "hinge"
        elif match:
            color = "pad" if int(match.group(1)) >= 3 else "shell"
            if "visual_" in path and path.rsplit("visual_", 1)[-1][:1] in ("1", "2"):
                color = "rail"
        else:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[color])
        count += 1
    print(f"COLORED_VISUAL_PRIMS {count}", flush=True)


def make_video(frames_dir: Path, mp4_path: Path, fps: int) -> None:
    frames = sorted(frames_dir.glob("rgb*.png"))
    if not frames:
        raise RuntimeError(f"no RGB frames in {frames_dir}")
    pattern = str(frames_dir / "rgb_%04d.png")
    command = ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4_path)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def opposition_pose(dof_names: list[str], limits: dict[str, tuple[float, float]], frame: int, frames: int) -> tuple[np.ndarray, str]:
    """Thumb-to-fingertip sequence: index, middle, ring, then little.

    This is a kinematic demonstration (not a force/contact simulation).  Each
    target receives a short hold; the other fingers remain comparatively open.
    """
    targets = ("index", "middle", "ring", "little")
    segment = max(1, frames // len(targets))
    target = targets[min(frame // segment, len(targets) - 1)]
    local = (frame % segment) / max(1, segment - 1)
    # Ease into the pose during the first third and preserve a visible hold.
    blend = min(1.0, local * 3.2)
    positions = np.zeros((1, len(dof_names)), dtype=np.float32)
    for i, name in enumerate(dof_names):
        lower, upper = limits[name]
        if name == "palm_to_thumb_cmc_pronation":
            # Sweep across the four target rays: index -> little.
            target_value = math.radians((18.0, 8.0, -10.0, -25.0)[targets.index(target)])
        elif name == "thumb_cmc_base_to_thumb_mc":
            target_value = math.radians(48.0)
        elif name == "thumb_mc_to_thumb_pp":
            target_value = math.radians(42.0)
        elif name == "thumb_pp_to_thumb_dp":
            target_value = math.radians(58.0)
        elif name.startswith(f"{target}_"):
            # Distal targets require progressively more flexion in this
            # simplified, straight-bone URDF to bring the fingertip palmward.
            if "mc_to" in name:
                target_value = math.radians(68.0)
            elif "pp_to" in name:
                target_value = math.radians(108.0)
            else:
                target_value = math.radians(72.0)
        else:
            target_value = math.radians(4.0) if "mc_to" in name else math.radians(2.0)
        target_value = min(max(target_value, lower + 0.03), upper - 0.03)
        positions[0, i] = lower + 0.03 + blend * (target_value - (lower + 0.03))
    return positions, target


def thumb_workspace_pose(dof_names: list[str], limits: dict[str, tuple[float, float]], frame: int, frames: int) -> tuple[np.ndarray, str]:
    """Move the radial-base thumb through abduction, opposition, then flexion."""
    keyframes = (
        ("open", 0.10, 0.08, 0.05, 0.05),
        ("abduct", 0.40, 0.08, 0.05, 0.05),
        ("opposition", 1.18, 0.08, 0.05, 0.05),
        ("pinch", 1.18, 0.42, 0.50, 0.40),
        ("open", 0.10, 0.08, 0.05, 0.05),
    )
    progress = frame / max(1, frames - 1) * (len(keyframes) - 1)
    left = min(int(progress), len(keyframes) - 2)
    fraction = progress - left
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    values = [keyframes[left][j] * (1 - blend) + keyframes[left + 1][j] * blend for j in range(1, 5)]
    thumb_names = (
        "link_1_thumb_spread_joint",
        "link_0_0_to_link_1_0",
        "link_1_0_to_link_1_1",
        "link_1_1_to_link_1_2",
    )
    positions = np.zeros((1, len(dof_names)), dtype=np.float32)
    for i, name in enumerate(dof_names):
        lower, upper = limits[name]
        value = values[thumb_names.index(name)] if name in thumb_names else max(lower, min(upper, 0.04))
        positions[0, i] = min(max(value, lower + 0.01), upper - 0.01)
    return positions, keyframes[left + 1][0]


def validate_one(urdf_path: Path, output_dir: Path, frames: int, fps: int, mode: str) -> dict:
    omni.usd.get_context().new_stage()
    setup_scene()
    limits = urdf_limits(urdf_path)
    prim_path, art = import_hand(urdf_path)
    color_imported_hand()
    # Physics views are created after the timeline is playing; this follows the
    # installed Isaac Sim URDF-import example's initialization sequence.
    omni.timeline.get_timeline_interface().play()
    APP.update()
    art.initialize()
    if not art.is_physics_handle_valid():
        raise RuntimeError(f"invalid articulation handle: {urdf_path.name}")
    dof_names = list(art.dof_names)
    active = [(i, limits[name]) for i, name in enumerate(dof_names) if name in limits]
    if len(active) != len(limits):
        raise RuntimeError(f"DOF mapping mismatch: imported={dof_names}, urdf={sorted(limits)}")

    model_dir = output_dir.resolve() / urdf_path.stem
    frames_dir = model_dir / "frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    settings = carb.settings.get_settings()
    settings.set("/omni/replicator/captureOnPlay", False)
    settings.set("/omni/replicator/asyncRendering", False)
    settings.set("/app/asyncRendering", False)
    camera = rep.create.camera(
        # Front/palm view: the four rays spread along Y and extend along Z,
        # therefore the palm face is normal to the X axis in this URDF.
        position=(-0.38, -0.29, 0.24),
        look_at=(0.025, 0.0, 0.07),
        clipping_range=(0.001, 10.0),
        focal_length=38.0,
        name="ValidationCamera",
    )
    render_product = rep.create.render_product(camera, (768, 768))
    for _ in range(5):
        APP.update()
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frames_dir.resolve()), rgb=True)
    writer.attach([render_product])
    measured_low = np.full(len(dof_names), np.inf)
    measured_high = np.full(len(dof_names), -np.inf)
    active_targets: list[str] = []
    for frame in range(frames):
        if mode == "thumb_workspace":
            positions, target = thumb_workspace_pose(dof_names, limits, frame, frames)
            active_targets.append(target)
        elif mode == "opposition":
            positions, target = opposition_pose(dof_names, limits, frame, frames)
            active_targets.append(target)
        else:
            phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * frame / max(1, frames - 1))
            positions = np.zeros((1, len(dof_names)), dtype=np.float32)
            for index, (lower, upper) in active:
                # Retain a margin from each hard stop so the sweep tests reachability without contact singularities.
                margin = min(0.06, 0.15 * (upper - lower))
                positions[0, index] = lower + margin + phase * (upper - lower - 2.0 * margin)
        art.set_joint_positions(positions)
        APP.update()
        actual = art.get_joint_positions()[0]
        measured_low = np.minimum(measured_low, actual)
        measured_high = np.maximum(measured_high, actual)
        rep.orchestrator.step(rt_subframes=4)
    writer.detach()
    render_product.destroy()
    rep.orchestrator.wait_until_complete()
    mp4_path = model_dir / f"{urdf_path.stem}_{mode}.mp4"
    make_video(frames_dir, mp4_path, fps)
    report = {
        "urdf": str(urdf_path),
        "articulation_root": prim_path,
        "dof_names": dof_names,
        "swept_revolute_dofs": len(active),
        "frames": frames,
        "mode": mode,
        "target_sequence": active_targets if mode in ("opposition", "thumb_workspace") else None,
        "video": str(mp4_path),
        "limits_rad": {name: list(limits[name]) for name in dof_names if name in limits},
        "measured_range_rad": {
            name: [float(measured_low[i]), float(measured_high[i])] for i, name in enumerate(dof_names) if name in limits
        },
    }
    (model_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    omni.timeline.get_timeline_interface().stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=("rom", "opposition", "thumb_workspace"), default="rom")
    args = parser.parse_args()
    reports = []
    for urdf in sorted(args.urdf_dir.glob("*.urdf")):
        print(f"VALIDATING {urdf.name}", flush=True)
        reports.append(validate_one(urdf, args.output_dir, args.frames, args.fps, args.mode))
    (args.output_dir / "summary.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print("VALIDATION_COMPLETE", flush=True)
    APP.close()


if __name__ == "__main__":
    main()
