"""Compile one bone genotype into a gated mechanical training phenotype."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from code_to_urdf import generate_urdf_from_dict
from collision_gate import audit_generated_urdf, validate_morphology
from compile_mechanical_hand import compile_hand
from mirror_mechanical_urdf import mirror


def fingerprint(hand):
    genotype = dict(hand)
    genotype.pop("evolution_id", None)
    payload = json.dumps(genotype, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def pipeline_fingerprint():
    digest = hashlib.sha256()
    for name in ("code_to_urdf.py", "compile_mechanical_hand.py", "mirror_mechanical_urdf.py",
                 "collision_gate.py", "mechanical_hand_interface.py",
                 "validate_collision_runtime.py", "mechanical_contact_model.py", "mechanical_contact_materials.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def run_gate(script, arguments, output, timeout=900):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "process.log").open("w") as log:
        import os
        result = subprocess.run([os.environ.get("EVOLUTION_MECHANICAL_PYTHON", sys.executable),
                                 str(Path(__file__).with_name(script)), *arguments,
                                 "--output_dir", str(output), "--headless", "--device", "cuda:0"],
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return result.returncode


def build_gated_phenotype(hand, root):
    root = Path(root).resolve() / fingerprint(hand)
    report_path = root / "phenotype_gate.json"
    pipeline = pipeline_fingerprint()
    if report_path.is_file():
        old = json.loads(report_path.read_text())
        if (old.get("passed") and old.get("fingerprint") == fingerprint(hand)
                and old.get("pipeline_sha256") == pipeline):
            return root, old
    root.mkdir(parents=True, exist_ok=True)
    report = {"fingerprint": fingerprint(hand), "pipeline_sha256": pipeline,
              "passed": False, "checks": {}}
    try:
        valid, topology = validate_morphology(hand)
        report["checks"]["topology"] = topology
        if not valid:
            raise ValueError("bone topology invalid")
        bone = root / "bone" / "urdf" / "current_agent.urdf"
        generate_urdf_from_dict(hand, str(bone.parent.parent / "meshes"), str(bone))
        assets = root / "assets"
        right_mechanical = root / "right_mechanical" / "urdf" / "current_agent.urdf"
        left_mechanical = root / "left_mechanical" / "urdf" / "current_agent.urdf"
        mapping = compile_hand(bone, right_mechanical)
        mirror(right_mechanical, left_mechanical)
        report["contact_model"] = mapping.get("contact_model")
        report["joint_count"] = mapping["actuated_joint_count"]
        report["rigid_palm"] = mapping["rigid_palm"]
        if not mapping["rigid_palm"] or mapping["rigid_palm"]["root_joint_count"] < 2:
            raise ValueError("Mechanical phenotype has no morphology-adaptive rigid palm")
        assets = root / "assets"
        for side, mechanical in (("right", right_mechanical), ("left", left_mechanical)):
            static_pass, static = audit_generated_urdf(mechanical)
            report["checks"][f"{side}_static"] = {"passed": static_pass, "reasons": static.get("reasons", [])}
            if not static_pass:
                raise ValueError(f"{side} mechanical collision mesh invalid")
            tree = ET.parse(mechanical).getroot()
            palm = tree.find("link[@name='link_palm_chassis']")
            if palm is None or not palm.findall("collision"):
                raise ValueError(f"{side} mechanical URDF lacks collidable rigid palm")
            (assets / side).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(mechanical.parent.parent, assets / side, dirs_exist_ok=True)
            runtime = root / f"{side}_runtime"
            code = run_gate("validate_collision_runtime.py",
                            ["--urdf", str(assets / side / "urdf" / "current_agent.urdf"),
                             "--exit_after_report"], runtime, 600)
            runtime_report = runtime / "runtime.json"
            if not runtime_report.is_file():
                raise RuntimeError(f"{side} runtime gate exited without runtime.json; inspect {runtime / 'process.log'}")
            physical = json.loads(runtime_report.read_text())
            report["checks"][f"{side}_runtime"] = {"exit_code": code, "passed": physical.get("passed", False)}
            if code or not physical.get("passed"):
                raise ValueError(f"{side} mechanical motion/collision gate failed")
            report["checks"][f"{side}_rigid_palm"] = {
                "passed": True, "collision_count": len(palm.findall("collision")),
                "jointed_to_fingers": True,
            }
        (root / "lineage.json").write_text(json.dumps({"lineage": {"candidate": {"urdf_info": hand}}}, indent=2))
        report["passed"] = True
        report["assets"] = str(assets)
    except Exception as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    finally:
        report_path.write_text(json.dumps(report, indent=2))
    return root, report
