"""Fail-closed Isaac collision screening, separate from task success."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from collision_gate import _topology_module


class CollisionRuntimeInfrastructureError(RuntimeError):
    """The screening could not run; do not label all candidates malformed."""


def validate_runtime_report(report, returncode, asset_fingerprint):
    if not report or report.get("completed") is not True or returncode not in (0, 2):
        raise CollisionRuntimeInfrastructureError("Isaac collision screening did not complete")
    if report.get("asset_fingerprint") != asset_fingerprint:
        raise CollisionRuntimeInfrastructureError("Runtime report does not match the audited meshes")
    required = ("passed", "samples", "self_collision_enabled", "effort_limits_match_config", "effort_limits_match_urdf",
                "peak_limit_error_rad", "dynamic_source_overlap_events",
                "missing_filters", "unexpected_filters", "colliders")
    if any(key not in report for key in required):
        raise CollisionRuntimeInfrastructureError("Incomplete collision screening schema")
    passed = (report["passed"] is True and returncode == 0
              and len(report["samples"]) >= 49
              and report["self_collision_enabled"] is True
              and report["effort_limits_match_config"] is True
              and report["effort_limits_match_urdf"] is True
              and 0 <= report["peak_limit_error_rad"] <= 0.005
              and not report["dynamic_source_overlap_events"]
              and not report["missing_filters"] and not report["unexpected_filters"]
              and bool(report["colliders"])
              and all(item.get("enabled") is True for item in report["colliders"])
              and all(sample.get("passed") is True for sample in report["samples"]))
    return bool(passed)


def run_isaac_collision_gate(urdf_path, output_dir, timeout_s=600):
    path = Path(urdf_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    asset_fingerprint = _topology_module().collision_asset_fingerprint(path)
    digest = hashlib.sha256(asset_fingerprint.encode())
    sources = [source_root / name for name in (
        "runtime_collision_gate.py", "validate_collision_runtime.py",
        "collision_gate.py", "isaaclab_tool.py")]
    sources += [source_root.parent / "evolution_tasks" / name
                for name in ("collision_topology.py", "hand_collision.py")]
    for source in sources:
        digest.update(source.read_bytes())
    signature = digest.hexdigest()
    status_path = output / "status.json"
    if status_path.exists():
        cached = json.loads(status_path.read_text())
        if cached.get("signature") == signature and cached.get("passed") is True:
            report_path = output / "runtime.json"
            if report_path.exists() and validate_runtime_report(
                json.loads(report_path.read_text()), 0, asset_fingerprint
            ):
                return True, cached
    report_path = output / "runtime.json"
    report_path.unlink(missing_ok=True)
    command = [sys.executable, str(source_root / "validate_collision_runtime.py"),
               "--headless", "--device", "cuda:0", "--urdf", str(path),
               "--output_dir", str(output), "--exit_after_report",
               "--kit_args=--/plugins/carb.tasking.plugin/threadCount=1"]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(source_root) + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env["OMP_NUM_THREADS"] = "1"
    log_path = output / "runtime.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=child_env, cwd=str(source_root.parent),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise CollisionRuntimeInfrastructureError(f"Collision audit timed out: {log_path}") from exc
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    passed = validate_runtime_report(report, returncode, asset_fingerprint)
    status = {"passed": passed, "signature": signature, "gate": "isaac_collision_v1",
              "asset_fingerprint": asset_fingerprint, "report_path": str(report_path),
              "log_path": str(log_path), "returncode": returncode,
              "reasons": [] if passed else ["runtime_collision_or_joint_constraint_failure"]}
    temporary_status = output / "status.json.tmp"
    temporary_status.write_text(json.dumps(status, indent=2))
    temporary_status.replace(status_path)
    return passed, status


def audit_morphology_in_isaac(urdf_info, audit_root):
    """Generate and check both hands without importing the training entrypoint."""
    from code_to_urdf import generate_urdf_from_dict
    from collision_gate import audit_generated_urdf
    from mirror_agent import create_mirror_hand
    from isaaclab_tool import parse_urdf_and_generate_articulation_cfg

    root = Path(audit_root)
    report = {"passed": True, "gate": "bilateral_collision_v1", "reasons": []}
    for side, morphology in (("right", urdf_info),
                              ("left", create_mirror_hand(urdf_info, "collision_gate_left"))):
        side_root = root / side
        urdf = side_root / "urdf" / "current_agent.urdf"
        generate_urdf_from_dict(morphology, str(side_root / "meshes"), str(urdf))
        passed, static = audit_generated_urdf(urdf)
        report[side] = {"static": static}
        if passed:
            try:
                parse_urdf_and_generate_articulation_cfg(
                    str(urdf), str(urdf), str(side_root / "compatibility_cfg.py")
                )
            except (ValueError, KeyError) as exc:
                passed = False
                report[side]["configuration"] = {"passed": False, "reason": str(exc)}
                report["reasons"].append(f"{side}_configuration_unadaptable:{exc}")
        if passed:
            passed, runtime = run_isaac_collision_gate(urdf, side_root / "runtime")
            report[side]["runtime"] = runtime
        if not passed:
            report["passed"] = False
            report["reasons"].append(f"{side}_collision_failure")
            break
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "bilateral.json.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(root / "bilateral.json")
    return report["passed"], report
