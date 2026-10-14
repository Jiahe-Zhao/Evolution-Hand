"""Exercise the production hand importer and collision filters in Isaac."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--depenetration_velocity", type=float, default=None)
parser.add_argument("--contact_offset", type=float, default=None)
parser.add_argument("--physics_dt", type=float, default=1 / 120)
parser.add_argument("--sample_interval", type=float, default=0.25)
parser.add_argument("--effort_cap", type=float, default=None)
parser.add_argument("--armature", type=float, default=None)
parser.add_argument("--isolate_joint", default=None)
parser.add_argument("--exit_after_report", action="store_true",
                    help="Exit the isolated audit process after flushing its completed report.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from collision_gate import audit_generated_urdf, _origin, _audit_neutral_overlap, _topology_module
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg


def main():
    started = time.monotonic()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = Path(args.urdf).resolve()
    static_ok, static = audit_generated_urdf(path)
    (output / "static.json").write_text(json.dumps(static, indent=2))
    if not static_ok:
        raise ValueError(static["reasons"])
    cfg_path = output / "hand_cfg.py"
    parse_urdf_and_generate_articulation_cfg(str(path), str(path), str(cfg_path))
    spec = importlib.util.spec_from_file_location("collision_runtime_cfg", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.CURRENT_HAND_CFG.replace(prim_path="/World/Hand")
    for actuator in cfg.actuators.values():
        if args.effort_cap is not None:
            actuator.effort_limit_sim = {key: min(value, args.effort_cap)
                                       for key, value in actuator.effort_limit_sim.items()}
        if args.armature is not None:
            actuator.armature = args.armature
    if args.depenetration_velocity is not None:
        cfg.spawn.rigid_props.max_depenetration_velocity = args.depenetration_velocity
    if args.contact_offset is not None:
        cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=args.contact_offset, rest_offset=0.0
        )
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.physics_dt, device=args.device))
    hand = Articulation(cfg)
    sim.reset()
    hand.update(sim.get_physics_dt())
    root_prim = sim.stage.GetPrimAtPath("/World/Hand")
    colliders, filters, modes = [], set(), {}
    authored_physics = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        attributes = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                      if a.GetName().startswith(("physxArticulation:", "physxCollision:", "physxRigidBody:"))}
        if attributes:
            authored_physics.append({"path": str(prim.GetPath()), "attributes": attributes})
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({"path": str(prim.GetPath()), "enabled": bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())})
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mode = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())
            modes[mode] = modes.get(mode, 0) + 1
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                filters.add(tuple(sorted((prim.GetName(), target.name))))
    expected_filters = set(_topology_module().collision_exclusions(path, hand.body_names))
    missing_filters = expected_filters - filters
    unexpected_filters = {p for p in filters - expected_filters if all(n in hand.body_names for n in p)}
    (output / "imported_physics.json").write_text(json.dumps({
        "attributes": authored_physics, "colliders": colliders, "filters": sorted(filters),
        "diagnostic_overrides": {"depenetration_velocity": args.depenetration_velocity,
                                 "contact_offset": args.contact_offset, "physics_dt": args.physics_dt,
                                 "effort_cap": args.effort_cap, "armature": args.armature},
    }, indent=2))
    sim.stage.Export(str(output / "loaded_stage.usda"))
    root = ET.parse(path).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    local_meshes, source_mapping = [], {}
    for link in root.findall("link"):
        name = link.get("name")
        body, transform = name, np.eye(4)
        while body not in hand.body_names:
            joint = parents[body]
            if joint.get("type") != "fixed":
                raise ValueError(f"Missing moving body {body}")
            transform = _origin(joint.find("origin")) @ transform
            body = joint.find("parent").get("link")
        source_mapping[name] = (hand.body_names.index(body), transform)
        for index, collision in enumerate(link.findall("collision")):
            node = collision.find("geometry/mesh")
            mesh = trimesh.load(path.parent / node.get("filename"), force="mesh", process=True)
            mesh.apply_scale(np.fromstring(node.get("scale", "1 1 1"), sep=" "))
            local_meshes.append((name, index, mesh, _origin(collision.find("origin"))))
    limits = hand.root_physx_view.get_dof_limits().to(hand.device)
    runtime_efforts = hand.root_physx_view.get_dof_max_forces()[0].cpu().numpy()
    source_efforts = {j.get("name"): float(j.find("limit").get("effort"))
                      for j in root.findall("joint") if j.get("type") != "fixed"}
    expected_efforts = np.array([source_efforts[name] for name in hand.joint_names])
    configured_efforts = np.minimum(expected_efforts, args.effort_cap) if args.effort_cap is not None else expected_efforts
    effort_limits_match = bool(np.allclose(runtime_efforts, configured_efforts, rtol=1e-5, atol=1e-6))
    q0 = torch.zeros_like(hand.data.joint_pos).clamp(limits[..., 0], limits[..., 1])
    target = limits[..., 1] * 0.8
    if args.isolate_joint is not None:
        if args.isolate_joint not in hand.joint_names:
            raise ValueError(f"Unknown isolated joint: {args.isolate_joint}")
        keep = hand.joint_names.index(args.isolate_joint)
        target[:] = 0.0
        target[:, keep] = limits[:, keep, 1] * 0.8
    for index, name in enumerate(hand.joint_names):
        if "thumb_spread" in name:
            target[:, index] = limits[:, index, 0] * 0.8
        elif "spread" in name:
            target[:, index] = 0
    hand.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    hand.set_joint_position_target(q0)
    sim.forward()
    hand.update(0.0)
    samples, peak_limit_error, peak_speed = [], 0.0, 0.0
    max_error_joint = None
    half_steps = round(6.0 / sim.get_physics_dt())
    sample_interval = max(1, round(args.sample_interval / sim.get_physics_dt()))
    for step in range(2 * half_steps + 1):
        # Smoothly close then open, without teleporting intermediate poses.
        fraction = step / half_steps if step <= half_steps else (2 * half_steps - step) / half_steps
        blend = fraction * fraction * (3 - 2 * fraction)
        hand.set_joint_position_target(q0 + blend * (target - q0))
        hand.write_data_to_sim()
        sim.step(render=False)
        hand.update(sim.get_physics_dt())
        q, v = hand.data.joint_pos, hand.data.joint_vel
        if not torch.isfinite(q).all() or not torch.isfinite(v).all():
            raise ValueError("Nonfinite joint state")
        errors = torch.maximum(limits[..., 0] - q, q - limits[..., 1]).clamp_min(0)
        error = float(errors.max())
        if error > peak_limit_error:
            peak_limit_error = error
            max_error_joint = hand.joint_names[int(errors[0].argmax())]
        peak_speed = max(peak_speed, float(v.abs().max()))
        if step % sample_interval == 0:
            positions = hand.data.body_pos_w[0].cpu().numpy()
            quats = hand.data.body_quat_w[0].cpu().numpy()
            poses = {}
            for name, (body_id, offset) in source_mapping.items():
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat(quats[body_id][[1, 2, 3, 0]]).as_matrix()
                matrix[:3, 3] = positions[body_id]
                poses[name] = matrix @ offset
            audit = _audit_neutral_overlap(path, root, local_meshes, world_poses=poses)
            samples.append({"step": step, "limit_error_rad": error, "passed": audit["passed"], "overlaps": audit["overlaps"],
                            "joint_positions_rad": q[0].cpu().tolist()})
            print("[COLLISION_SWEEP]", json.dumps(samples[-1]), flush=True)
    dynamic_overlap_events = [
        overlap
        for sample in samples
        for overlap in sample["overlaps"]
        if overlap.get("severe")
    ]
    actual_self_collision = any(
        item["attributes"].get("physxArticulation:enabledSelfCollisions") == "True"
        for item in authored_physics
    )
    passed = (bool(colliders) and all(c["enabled"] for c in colliders) and not missing_filters
              and not unexpected_filters and peak_limit_error <= 0.005
              and not dynamic_overlap_events and actual_self_collision and effort_limits_match)
    report = {"passed": passed, "isolated_joint": args.isolate_joint, "scope": f"One coordinated close/open sweep, source-solid witness checks at {len(samples)} poses; not exhaustive continuous ROM or a reconstruction of cooked convex hulls.",
              "colliders": colliders, "approximation_modes": modes, "expected_filters": sorted(expected_filters),
              "missing_filters": sorted(missing_filters), "unexpected_filters": sorted(unexpected_filters),
              "self_collision_enabled": actual_self_collision,
              "effort_limits_match_config": effort_limits_match,
              "effort_limits_match_urdf": bool(np.allclose(runtime_efforts, expected_efforts)),
              "runtime_effort_limits": dict(zip(hand.joint_names, runtime_efforts.tolist())),
              "peak_limit_error_rad": peak_limit_error, "worst_joint": max_error_joint, "peak_speed_rad_s": peak_speed,
              "dynamic_source_overlap_events": dynamic_overlap_events,
              "samples": samples, "elapsed_s": time.monotonic() - started}
    report["completed"] = True
    report["asset_fingerprint"] = static["asset_fingerprint"]
    temporary_report = output / "runtime.json.tmp"
    temporary_report.write_text(json.dumps(report, indent=2))
    temporary_report.replace(output / "runtime.json")
    print("[COLLISION_RUNTIME_RESULT]", json.dumps({k:v for k,v in report.items() if k not in {"colliders", "samples", "expected_filters"}}), flush=True)
    return passed


try:
    passed = main()
    if args.exit_after_report:
        # Only for this isolated, non-training process. Some headless Kit
        # builds hang in plugin teardown after a completed simulation.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if passed else 2)
finally:
    app.close(wait_for_replicator=False)
sys.exit(0 if passed else 2)
