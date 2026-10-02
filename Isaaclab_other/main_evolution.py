"""主要的进化入口，支持外层与内层断点续跑，以及同代个体并行评估。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import uuid
import copy

import numpy as np

from class_population import Lineage
from evaluation_interface import close_evaluation_workers, evaluation
from policy_inheritance import select_parent_checkpoint
from variation import choose_target, seed_initial_population, variation
from collision_gate import audit_generated_urdf, lightweight_geometry_prefilter, validate_morphology
from runtime_collision_gate import CollisionRuntimeInfrastructureError, audit_morphology_in_isaac


HOME_DIR = os.path.expanduser("~")
EVOLUTION_ROOT = os.environ.get("EVOLUTION_ROOT", os.path.join(HOME_DIR, "Evolution_PC"))
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", os.path.join(HOME_DIR, "IsaacLab"))
ISAACLAB_TASK_ROOT = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "evolution_tasks"
)
ISAACLAB_OTHER_ROOT = os.path.join(EVOLUTION_ROOT, "Isaaclab_other")
EVOLUTION_LOG_ROOT = os.path.join(EVOLUTION_ROOT, "evolution_tasks", "logs", "evolution_task")


def _env_int(name, default):
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def _env_float(name, default):
    raw_value = os.environ.get(name)
    if raw_value in (None, ""):
        return default
    return float(raw_value)


def _env_flag(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_local_path(file_stem, suffix):
    if os.path.isabs(file_stem):
        return f"{file_stem}{suffix}"
    return os.path.join(ISAACLAB_OTHER_ROOT, f"{file_stem}{suffix}")


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _make_deterministic_seed(experiment_name, generation, individual, trial):
    seed_source = f"{experiment_name}:{generation}:{individual}:{trial}"
    return int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:8], 16)


def _make_child_id(experiment_name, generation, individual, trial):
    seed_source = f"{experiment_name}:{generation}:{individual}:{trial}:child"
    return uuid.uuid5(uuid.NAMESPACE_DNS, seed_source).hex


def _make_elite_id(experiment_name, generation, parent_id):
    seed_source = f"{experiment_name}:{generation}:elite:{parent_id}"
    return uuid.uuid5(uuid.NAMESPACE_DNS, seed_source).hex


def _run_generated_collision_gate(urdf_info, audit_root):
    """Generate and audit both hands, reusing a passed morphology result."""
    cache_root = os.environ.get(
        "EVOLUTION_COLLISION_CACHE_ROOT",
        os.path.join(ISAACLAB_OTHER_ROOT, "collision_gate_cache"),
    )
    os.makedirs(cache_root, exist_ok=True)
    payload = json.dumps(urdf_info, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{fingerprint}.json")
    cached = _load_json(cache_path)
    if cached and cached.get("passed") is True and cached.get("report"):
        report = copy.deepcopy(cached["report"])
        report["cache_hit"] = True
        return True, report
    passed, report = audit_morphology_in_isaac(urdf_info, audit_root)
    report["cache_hit"] = False
    _atomic_write_json(cache_path, {"fingerprint": fingerprint, "passed": bool(passed), "report": report})
    return passed, report


def _run_lightweight_prefilter(urdf_info):
    return lightweight_geometry_prefilter(urdf_info)


def _load_or_initialize_lineage(
    experiment_json_path,
    check_point,
    initial_population_size,
    initial_population_attempts,
    initial_population_variation,
    initial_population_length,
    force_new_lineage,
):
    lineage = Lineage()
    if os.path.exists(experiment_json_path) and not force_new_lineage:
        lineage.load_from_file(experiment_json_path)
        if lineage.lineage:
            return lineage

    if check_point == "human":
        from human_hand_agent import initial_agent_hand
    elif check_point == "gorilla":
        from gorilla_hand_agent import initial_agent_hand
    elif check_point == "arboreal_prior":
        from arboreal_hand_agent import initial_agent_hand
    else:
        raise ValueError(f"Unsupported check_point: {check_point}")

    initial_audit_root = os.path.join(
        ISAACLAB_OTHER_ROOT,
        f"{os.path.basename(experiment_json_path)}_initial_collision_gate",
    )
    initial_reports_root = os.path.join(
        ISAACLAB_OTHER_ROOT,
        f"{os.path.basename(experiment_json_path)}_initial_morphology_reports",
    )
    candidate_count = 0
    accepted_reports = []

    def accept_initial_candidate(urdf):
        nonlocal candidate_count
        index = candidate_count
        candidate_count += 1
        passed, report = validate_morphology(urdf)
        generated_report = None
        if passed:
            prefilter_passed, prefilter_report = _run_lightweight_prefilter(urdf)
            generated_report = {"prefilter": prefilter_report}
            if not prefilter_passed:
                passed = False
                generated_report["passed"] = False
                generated_report["reasons"] = prefilter_report["reasons"]
        if passed:
            candidate_root = os.path.join(initial_audit_root, f"candidate_{index:03d}")
            try:
                generated_passed, generated_report = _run_generated_collision_gate(
                    urdf, candidate_root
                )
            except CollisionRuntimeInfrastructureError:
                raise
            except Exception as exc:
                generated_passed = False
                generated_report = {
                    "passed": False,
                    "gate": "generated_urdf_v1",
                    "reasons": [f"generation_or_audit:{exc}"],
                }
            passed = generated_passed
        combined_report = {
            "source_morphology": report,
            "generated_collision": generated_report,
            "candidate_index": index,
        }
        _atomic_write_json(
            os.path.join(initial_reports_root, f"candidate_{index:03d}.json"),
            combined_report,
        )
        if passed:
            accepted_reports.append(combined_report)
        else:
            reasons = report.get("reasons", [])
            if generated_report:
                reasons = generated_report.get("reasons", reasons)
            print(f"[MORPHOLOGY_GATE] Rejecting initial morphology: {reasons}")
        return passed

    valid_initial_population = seed_initial_population(
        initial_agent_hand,
        population_size=initial_population_size,
        include_base=True,
        max_attempts=initial_population_attempts,
        standard_variation=initial_population_variation,
        standard_length=initial_population_length,
        candidate_validator=accept_initial_candidate,
    )
    if len(valid_initial_population) < initial_population_size:
        raise RuntimeError(
            f"Only {len(valid_initial_population)}/{initial_population_size} initial morphologies "
            "passed the morphology/collision gate. Increase generation attempts or repair the prior."
        )
    for idx, urdf in enumerate(valid_initial_population[:initial_population_size]):
        new_id = uuid.uuid4().hex
        urdf["evolution_id"] = new_id
        lineage.add_individual(-1, idx, urdf, 0, new_id, metadata={
            "seed_stage": "initial_population", "collision_gate": accepted_reports[idx],
        })
    lineage.save_to_file(experiment_json_path)
    return lineage


def _runtime_state_path(experiment_name):
    return _resolve_local_path(experiment_name, "_runtime_state.json")


def _legacy_evaluation_state_path(experiment_name):
    return _resolve_local_path(experiment_name, "_evaluation_state.json")


def _evaluation_state_dir(experiment_name):
    return _resolve_local_path(experiment_name, "_evaluation_states")


def _evaluation_state_path_for_child(experiment_name, child_id):
    return os.path.join(_evaluation_state_dir(experiment_name), f"{child_id}.json")


def _build_runtime_state(
    generation,
    current_individual=None,
    trial=0,
    phase="ready",
    pending_children=None,
    parallel_slots=1,
):
    ordered_pending = sorted(
        pending_children or [],
        key=lambda item: (item["generation"], item["individual"], item["trial"], item["child_id"]),
    )
    pending_child = ordered_pending[0] if len(ordered_pending) == 1 else None
    return {
        "version": 2,
        "current_generation": generation,
        "current_individual": current_individual,
        "trial": trial,
        "phase": phase,
        "pending_child": pending_child,
        "pending_children": ordered_pending,
        "parallel_slots": parallel_slots,
    }


def _save_runtime_state(path, state):
    _atomic_write_json(path, state)
    return state


def _normalize_pending_child(child, generation, individual, trial):
    normalized = dict(child)
    normalized["generation"] = generation
    normalized["individual"] = individual
    normalized["trial"] = trial
    normalized.setdefault("slot_id", 0)
    return normalized


def _load_runtime_state(path, parallel_slots):
    state = _load_json(path)
    if not state:
        return None
    version = state.get("version")
    if version == 2:
        return _build_runtime_state(
            state["current_generation"],
            current_individual=state.get("current_individual"),
            trial=state.get("trial", 0),
            phase=state.get("phase", "ready"),
            pending_children=state.get("pending_children", []),
            parallel_slots=state.get("parallel_slots", parallel_slots),
        )
    if version == 1:
        pending_children = []
        if state.get("pending_child") is not None:
            pending_children.append(
                _normalize_pending_child(
                    state["pending_child"],
                    state["current_generation"],
                    state.get("current_individual"),
                    state.get("trial", 0),
                )
            )
        return _build_runtime_state(
            state["current_generation"],
            current_individual=state.get("current_individual"),
            trial=state.get("trial", 0),
            phase=state.get("phase", "ready"),
            pending_children=pending_children,
            parallel_slots=parallel_slots,
        )
    return None


def _next_progress_marker(pending_children):
    if not pending_children:
        return None, 0
    next_child = min(pending_children, key=lambda item: (item["individual"], item["trial"], item["child_id"]))
    return next_child["individual"], next_child["trial"]


def _persist_generation_state(path, generation, pending_children, parallel_slots):
    current_individual, trial = _next_progress_marker(pending_children)
    phase = "evaluating" if pending_children else "ready"
    return _save_runtime_state(
        path,
        _build_runtime_state(
            generation,
            current_individual=current_individual,
            trial=trial,
            phase=phase,
            pending_children=pending_children,
            parallel_slots=parallel_slots,
        ),
    )


def _build_child_entry(
    experiment_name,
    generation,
    individual,
    trial,
    current_urdf,
    variation_probabilities,
    pending_child=None,
):
    if pending_child is not None:
        return _normalize_pending_child(pending_child, generation, individual, trial)

    deterministic_seed = _make_deterministic_seed(experiment_name, generation, individual, trial)
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    link_code, task_code, strength = choose_target(current_urdf, variation_probabilities)
    print("link_code, task_code, strength:", link_code, task_code, strength)
    success_tag, new_urdf = variation(
        current_urdf,
        link_code,
        task_code,
        strength,
        standard_variation=variation_standard,
        standard_length=variation_length,
    )
    metadata = {
        "trial": trial,
        "seed": deterministic_seed,
        "link_code": link_code,
        "task_code": task_code,
        "strength": strength,
    }
    child_id = _make_child_id(experiment_name, generation, individual, trial)
    print("success_tag, new_urdf:", success_tag)
    print(new_urdf)
    if not success_tag:
        return None

    new_urdf["evolution_id"] = child_id
    return {
        "generation": generation,
        "individual": individual,
        "trial": trial,
        "child_id": child_id,
        "urdf_info": new_urdf,
        "metadata": metadata,
        "slot_id": 0,
    }


def _assign_slots(children, parallel_slots):
    assigned = []
    for index, child in enumerate(
        sorted(children, key=lambda item: (item["generation"], item["individual"], item["trial"], item["child_id"]))
    ):
        normalized = dict(child)
        normalized["slot_id"] = index % parallel_slots
        assigned.append(normalized)
    return assigned


def _collect_pending_children(
    runtime_state,
    current_generation,
    surviving_individuals,
    hand_lineage,
    experiment_name,
    max_variation,
    max_variation_attempts,
    variation_probabilities,
):
    pending_lookup = {}
    if runtime_state["phase"] == "evaluating" and runtime_state.get("pending_children"):
        for pending in runtime_state["pending_children"]:
            if pending["generation"] != current_generation or hand_lineage.has_individual_id(pending["child_id"]):
                continue
            normalized = _normalize_pending_child(
                pending, current_generation, pending["individual"], pending["trial"]
            )
            pending_lookup[(normalized["individual"], normalized["trial"])] = normalized
    if runtime_state.get("pending_child") is not None and runtime_state["phase"] == "evaluating":
        pending_child = _normalize_pending_child(
            runtime_state["pending_child"],
            current_generation,
            runtime_state.get("current_individual"),
            runtime_state.get("trial", 0),
        )
        pending_lookup[(pending_child["individual"], pending_child["trial"])] = pending_child

    children = []
    for current_individual in surviving_individuals:
        if (
            runtime_state["current_generation"] == current_generation
            and runtime_state["current_individual"] is not None
            and current_individual < runtime_state["current_individual"]
        ):
            continue

        start_trial = 0
        if (
            runtime_state["current_generation"] == current_generation
            and runtime_state["current_individual"] == current_individual
        ):
            start_trial = runtime_state["trial"]

        current_urdf = hand_lineage.lineage[(current_generation, current_individual)]["urdf_info"]
        valid_children = 0
        trial = start_trial
        # Invalid geometry must not silently shrink a generation.  The target
        # remains `max_variation` valid children per parent, with a finite
        # deterministic retry budget to avoid an unbounded mutation loop.
        while valid_children < max_variation and trial < max_variation_attempts:
            pending_child = pending_lookup.get((current_individual, trial))
            child = _build_child_entry(
                experiment_name,
                current_generation,
                current_individual,
                trial,
                current_urdf,
                variation_probabilities,
                pending_child=pending_child,
            )
            trial += 1
            if child is None:
                continue
            if hand_lineage.has_individual_id(child["child_id"]):
                continue
            gate_passed, gate_report = validate_morphology(child["urdf_info"])
            child["metadata"] = dict(child.get("metadata", {}))
            child["metadata"]["morphology_gate"] = gate_report
            _atomic_write_json(
                os.path.join(
                    ISAACLAB_OTHER_ROOT,
                    f"{os.path.basename(experiment_name)}_morphology_reports",
                    f"{child['child_id']}.json",
                ),
                {"individual_id": child["child_id"], "generation": child["generation"],
                 "trial": child["trial"], "report": gate_report},
            )
            if not gate_passed:
                print(
                    f"[MORPHOLOGY_GATE] Rejecting malformed child {child['child_id']}: "
                    f"{gate_report['reasons']}"
                )
                continue
            prefilter_passed, prefilter_report = _run_lightweight_prefilter(child["urdf_info"])
            child["metadata"]["lightweight_geometry_prefilter"] = prefilter_report
            if not prefilter_passed:
                print(
                    f"[PREFILTER] Rejecting child {child['child_id']}: "
                    f"{prefilter_report['reasons']}"
                )
                continue
            # Generate and inspect the actual collision assets before launching
            # Isaac. This catches malformed meshes produced by a mutation even
            # when the source-level topology remains legal.
            generated_gate_passed = True
            generated_gate_report = {"passed": True, "gate": "generated_urdf_v1"}
            try:
                generated_root = os.path.join(
                    ISAACLAB_OTHER_ROOT,
                    f"{os.path.basename(experiment_name)}_collision_gate",
                    child["child_id"],
                )
                generated_gate_passed, generated_gate_report = _run_generated_collision_gate(
                    child["urdf_info"], generated_root
                )
            except CollisionRuntimeInfrastructureError:
                raise
            except Exception as exc:
                generated_gate_passed = False
                generated_gate_report = {
                    "passed": False,
                    "gate": "generated_urdf_v1",
                    "reasons": [f"generation_or_audit:{exc}"],
                }
            child["metadata"]["generated_collision_gate"] = generated_gate_report
            _atomic_write_json(
                os.path.join(ISAACLAB_OTHER_ROOT, f"{os.path.basename(experiment_name)}_morphology_reports", f"{child['child_id']}.json"),
                {"individual_id": child["child_id"], "generation": child["generation"],
                 "trial": child["trial"], "source": gate_report, "generated": generated_gate_report},
            )
            if not generated_gate_passed:
                print(
                    f"[COLLISION_GATE] Rejecting child {child['child_id']}: "
                    f"{generated_gate_report.get('reasons', [])}"
                )
                continue
            if _env_flag("EVOLUTION_SCRIPTED_PREFLIGHT", True):
                passed, preflight = _run_scripted_preflight(child, experiment_name)
                child["metadata"] = dict(child.get("metadata", {}))
                child["metadata"]["scripted_preflight"] = preflight
                if not passed:
                    if _env_flag("EVOLUTION_REQUIRE_SCRIPTED_PREFLIGHT_SUCCESS", False):
                        print(f"[WARN] Rejecting child {child['child_id']} before RL: scripted preflight failed.")
                        continue
                    print(
                        f"[WARN] Scripted preflight failed for {child['child_id']}; "
                        "recording diagnostics and continuing with pure RL."
                    )
            children.append(child)
            valid_children += 1

    return children


def _migrate_legacy_evaluation_state(legacy_path, child_state_path, child_id):
    if os.path.exists(child_state_path):
        return
    legacy_state = _load_json(legacy_path)
    if not legacy_state or legacy_state.get("individual_id") != child_id:
        return
    _atomic_write_json(child_state_path, legacy_state)


def _task_slug(task_name):
    return (
        task_name.replace("Isaac-", "")
        .replace("-v0", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _make_task_run_name(experiment_name, individual_id, task_name, curriculum_stage=None):
    stage_suffix = f"_{curriculum_stage.lower()}" if curriculum_stage else ""
    return f"{experiment_name}_{individual_id[:8]}_{_task_slug(task_name)}{stage_suffix}"


def _remove_path(path):
    if not os.path.exists(path):
        return False
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return True
    except OSError as error:
        print(f"[WARN] Failed to remove artifact {path}: {error}")
        return False


def _cleanup_eliminated_children(experiment_name, generation, hand_lineage, ordered_tasks):
    removed_run_dirs = 0
    removed_state_files = 0
    for (gen, _individual_number), individual in hand_lineage.lineage.items():
        if gen != generation or individual.get("tag") != "eliminated":
            continue

        child_id = individual.get("id")
        if not child_id:
            continue

        child_state_path = _evaluation_state_path_for_child(experiment_name, child_id)
        child_state = _load_json(child_state_path) or {}
        run_names = dict(child_state.get("run_names", {}))

        run_names_to_remove = {
            run_name for run_key, run_name in run_names.items()
            if not run_key.startswith("stage2:")
        }
        for task_name in ordered_tasks:
            # Clean both curriculum stages, plus the legacy unqualified name.
            run_names_to_remove.add(_make_task_run_name(experiment_name, child_id, task_name))
            run_names_to_remove.add(_make_task_run_name(experiment_name, child_id, task_name, "stage1"))

        for run_name in run_names_to_remove:
            if _remove_path(os.path.join(EVOLUTION_LOG_ROOT, run_name)):
                removed_run_dirs += 1

        if _remove_path(child_state_path):
            removed_state_files += 1

    if removed_run_dirs or removed_state_files:
        print(
            f"[INFO] Cleaned eliminated children for generation {generation}: "
            f"run_dirs={removed_run_dirs}, state_files={removed_state_files}"
        )


def _run_scripted_preflight(child, experiment_name):
    """Run and persist physical scripted diagnostics before PPO training."""
    safe_experiment = os.path.basename(experiment_name.rstrip(os.sep))
    root = os.path.join(
        EVOLUTION_ROOT, "evolution_tasks", "logs", "preflight",
        f"{safe_experiment}_{child['child_id'][:8]}",
    )
    scripts = {
        "grasp": os.path.join(EVOLUTION_ROOT, "evolution_tasks", "task_grasp", "scripted_adaptive_grasp.py"),
        "branch": os.path.join(EVOLUTION_ROOT, "evolution_tasks", "task_suite", "scripted_adaptive_task_demo.py"),
        "forage": os.path.join(EVOLUTION_ROOT, "evolution_tasks", "task_suite", "scripted_adaptive_task_demo.py"),
        "strike": os.path.join(EVOLUTION_ROOT, "evolution_tasks", "task_suite", "scripted_adaptive_task_demo.py"),
    }
    signature_sources = set(scripts.values()) | {
        os.path.join(ISAACLAB_OTHER_ROOT, name)
        for name in ("code_to_urdf.py", "collision_gate.py", "isaaclab_tool.py")
    } | {
        os.path.join(EVOLUTION_ROOT, "evolution_tasks", name)
        for name in ("collision_topology.py", "hand_collision.py", "sphere_mesh_audit.py")
    } | {
        os.path.join(EVOLUTION_ROOT, "evolution_tasks", task_dir, filename)
        for task_dir, filename in (
            ("task_grasp", "evolution_grasp_env.py"),
            ("task_grasp", "evolution_grasp_env_cfg.py"),
            ("task_branch_grasp", "branch_grasp_env.py"),
            ("task_branch_grasp", "branch_grasp_env_cfg.py"),
            ("task_forage", "forage_env.py"),
            ("task_forage", "forage_env_cfg.py"),
            ("task_strike", "evolution_strike_env.py"),
            ("task_strike", "evolution_strike_env_cfg.py"),
        )
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(child["urdf_info"], sort_keys=True).encode("utf-8"))
    for source_path in sorted(signature_sources):
        digest.update(source_path.encode("utf-8"))
        with open(source_path, "rb") as source_file:
            digest.update(source_file.read())
    preflight_signature = digest.hexdigest()
    status_path = os.path.join(root, "status.json")
    cached = _load_json(status_path)
    if (
        cached
        and cached.get("passed") is True
        and cached.get("preflight_signature") == preflight_signature
    ):
        return True, cached

    os.makedirs(root, exist_ok=True)
    individual_key = "0_0"
    lineage_path = os.path.join(root, "candidate.json")
    _atomic_write_json(
        lineage_path,
        {"lineage": {individual_key: {"urdf_info": child["urdf_info"]}}},
    )
    task_results = {}
    timeout = _env_int("EVOLUTION_SCRIPTED_PREFLIGHT_TIMEOUT", 900)
    child_env = os.environ.copy()
    child_env["EVOLUTION_CODE_ROOT"] = EVOLUTION_ROOT
    for task_name, script_path in scripts.items():
        task_root = os.path.join(root, task_name)
        os.makedirs(task_root, exist_ok=True)
        metrics_path = os.path.join(task_root, "metrics.json")
        output_path = os.path.join(task_root, "unused.mp4")
        command = [sys.executable, script_path]
        if task_name != "grasp":
            command.append(task_name)
        command.extend(
            [
                "--lineage_json", lineage_path,
                "--individual_key", individual_key,
                "--output", output_path,
                "--metrics", metrics_path,
                "--preflight", "--headless",
            ]
        )
        if task_name == "grasp":
            # A force-based scripted success is not enough: reject/record
            # trajectories that penetrate the generated collision solids.
            command.append("--audit_mesh")
        if task_name != "grasp":
            command.extend(["--min_video_steps", "1"])
        log_path = os.path.join(task_root, "preflight.log")
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=EVOLUTION_ROOT,
                    env=child_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            metrics = _load_json(metrics_path) or {}
            physical_success = metrics.get(
                "success",
                metrics.get("environment_m3_success", False)
                or metrics.get("calibration_sustained_success", False),
            )
            passed = result.returncode == 0 and bool(physical_success)
            task_results[task_name] = {
                "passed": passed,
                "returncode": result.returncode,
                "metrics_path": metrics_path,
                "log_path": log_path,
            }
        except subprocess.TimeoutExpired:
            task_results[task_name] = {
                "passed": False,
                "error": f"timeout_after_{timeout}s",
                "log_path": log_path,
            }
        if not task_results[task_name]["passed"]:
            break

    status = {
        "individual_id": child["child_id"],
        "preflight_signature": preflight_signature,
        "passed": len(task_results) == len(scripts) and all(
            result["passed"] for result in task_results.values()
        ),
        "tasks": task_results,
    }
    _atomic_write_json(status_path, status)
    return status["passed"], status


# 基本配置
# Treat EVOLUTION_MAX_GENERATION as the total number of parent generations,
# starting from generation 0.
max_generation = _env_int("EVOLUTION_MAX_GENERATION", 1000)
max_population = _env_int("EVOLUTION_MAX_POPULATION", 1000)
max_variation = _env_int("EVOLUTION_MAX_VARIATION", 10)
retain_parent_elites = _env_flag("EVOLUTION_RETAIN_PARENT_ELITES", False)
restart_workers_at_boundaries = _env_flag("EVOLUTION_ISAAC_RESTART_AT_BOUNDARIES", False)
max_variation_attempts = max(
    max_variation,
    _env_int("EVOLUTION_MAX_VARIATION_ATTEMPTS", max_variation * 10),
)
parallel_slots = max(1, _env_int("EVOLUTION_PARALLEL_SLOTS", 1))
initial_population_size = _env_int("EVOLUTION_INITIAL_POPULATION_SIZE", 8)
initial_population_attempts = _env_int("EVOLUTION_INITIAL_POPULATION_ATTEMPTS", 200)
initial_population_variation = _env_float("EVOLUTION_INITIAL_POPULATION_VARIATION", 0.05)
initial_population_length = _env_float("EVOLUTION_INITIAL_POPULATION_LENGTH", 0.02)
variation_probabilities = {
    "change_link_length": 0.30,
    "change_link_radius": 0.10,
    "remove_link": 0.05,
    "add_link": 0.05,
    "change_joint_origin_translation": 0.05,
    "change_joint_origin_rpy": 0.05,
    "change_thumb_length": 0.30,
    "change_palm_curvature": 0.10,
}
experiment_save_path = os.environ.get("EVOLUTION_EXPERIMENT_NAME", "exp_20260709_multitask_1")
evaluation_tasks_env = os.environ.get("EVOLUTION_TASKS")
evaluation_taks = (
    {task.strip() for task in evaluation_tasks_env.split(",") if task.strip()}
    if evaluation_tasks_env
    else {
        "Isaac-EvolutionHand-Grasp-v0",
        "Isaac-EvolutionHand-BranchGrasp-v0",
        "Isaac-EvolutionHand-Forage-v0",
        "Isaac-EvolutionHand-Strike-v0",
    }
)
check_point = os.environ.get("EVOLUTION_INITIAL_AGENT", "arboreal_prior")
force_new_lineage = _env_flag("EVOLUTION_FORCE_NEW_LINEAGE", False)
inner_max_iterations_env = os.environ.get("ISAACLAB_MAX_ITERATIONS")
inner_max_iterations = int(inner_max_iterations_env) if inner_max_iterations_env else None
stage1_max_iterations = _env_int(
    "EVOLUTION_STAGE1_MAX_ITERATIONS",
    inner_max_iterations if inner_max_iterations is not None else 200,
)
stage2_max_iterations = _env_int("EVOLUTION_STAGE2_MAX_ITERATIONS", 500)
stage2_top_fraction = min(1.0, max(0.0, _env_float("EVOLUTION_STAGE2_TOP_FRACTION", 0.2)))
stage2_enabled = stage2_max_iterations > stage1_max_iterations and stage2_top_fraction > 0.0
single_stage_name = os.environ.get("EVOLUTION_SINGLE_STAGE_NAME", "stage1").lower()
if single_stage_name not in {"stage1", "stage2"}:
    raise ValueError("EVOLUTION_SINGLE_STAGE_NAME must be 'stage1' or 'stage2'")
variation_standard = _env_float("EVOLUTION_STANDARD_VARIATION", 0.2)
variation_length = _env_float("EVOLUTION_STANDARD_LENGTH", 0.1)

isaaclab_urdf_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab", "urdf", "current_agent.urdf")
isaaclab_urdf_mesh_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab", "mesh")
isaaclab_urdf_code_path = os.path.join(
    ISAACLAB_TASK_ROOT, "current_right_hand", "current_right_hand_cfg.py"
)
isaaclab_env_code_path = os.path.join(ISAACLAB_TASK_ROOT, "task_stone", "evolution_stone_grind_env_cfg.py")
isaaclab_test_result_path = EVOLUTION_LOG_ROOT

isaaclab_mirror_urdf_path = os.path.join(
    ISAACLAB_OTHER_ROOT, "agent_for_isaaclab_mirror", "urdf", "current_agent.urdf"
)
isaaclab_mirror_urdf_mesh_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab_mirror", "mesh")
isaaclab_mirror_urdf_code_path = os.path.join(
    ISAACLAB_TASK_ROOT, "current_left_hand", "current_left_hand_cfg.py"
)

experiment_json_path = _resolve_local_path(experiment_save_path, ".json")
runtime_state_json_path = _runtime_state_path(experiment_save_path)
legacy_evaluation_state_json_path = _legacy_evaluation_state_path(experiment_save_path)

if force_new_lineage and (
    os.path.exists(experiment_json_path)
    or os.path.exists(runtime_state_json_path)
    or os.path.isdir(_evaluation_state_dir(experiment_save_path))
):
    print(
        f"[WARN] Existing experiment state detected for {experiment_save_path}; overriding EVOLUTION_FORCE_NEW_LINEAGE=0 for safe resume."
    )
    force_new_lineage = False

hand_lineage = _load_or_initialize_lineage(
    experiment_json_path,
    check_point,
    initial_population_size,
    initial_population_attempts,
    initial_population_variation,
    initial_population_length,
    force_new_lineage,
)

runtime_state = _load_runtime_state(runtime_state_json_path, parallel_slots)
if runtime_state is None:
    start_generation = max(hand_lineage.get_max_generation(), 0)
    runtime_state = _save_runtime_state(
        runtime_state_json_path,
        _build_runtime_state(start_generation, phase="ready", parallel_slots=parallel_slots),
    )


for current_generation in range(runtime_state["current_generation"], max_generation):
    print(f"Generation {current_generation}: Starting mutation and evaluation.")
    surviving_individuals = sorted(hand_lineage.get_surviving_individuals_in_generation(current_generation))
    if not surviving_individuals:
        raise RuntimeError(
            f"Generation {current_generation} has no surviving individuals; "
            "cannot continue evolution. Inspect task failures and lineage before resuming."
        )

    pending_children = _collect_pending_children(
        runtime_state,
        current_generation,
        surviving_individuals,
        hand_lineage,
        experiment_save_path,
        max_variation,
        max_variation_attempts,
        variation_probabilities,
    )
    if pending_children:
        pending_children = _assign_slots(pending_children, parallel_slots)
        def _evaluate_stage(stage_children, stage_max_iterations, stage_name):
            if not stage_children:
                return {}

            _persist_generation_state(
                runtime_state_json_path,
                current_generation,
                stage_children,
                parallel_slots,
            )

            state_lock = threading.Lock()
            pending_by_id = {child["child_id"]: dict(child) for child in stage_children}
            stage_results = {}

            task_names = sorted(evaluation_taks)
            final_task_name = task_names[-1]

            def _run_slot_queue(slot_id, slot_children, batch_task):
                for child in slot_children:
                    child_state_path = _evaluation_state_path_for_child(experiment_save_path, child["child_id"])
                    _migrate_legacy_evaluation_state(
                        legacy_evaluation_state_json_path,
                        child_state_path,
                        child["child_id"],
                    )
                    evaluation_error = None
                    parent = hand_lineage.lineage.get((child['generation'], child['individual']), {})
                    parent_state = (_load_json(_evaluation_state_path_for_child(
                        experiment_save_path, parent['id'])) or {}) if parent.get('id') else {}
                    inherited_checkpoint = None
                    if _env_flag('EVOLUTION_INHERIT_POLICY', True) and stage_name == 'stage1':
                        inherited_checkpoint = select_parent_checkpoint(
                            parent, batch_task, EVOLUTION_LOG_ROOT, parent_state)
                    try:
                        current_score = evaluation(
                            child["urdf_info"],
                            [batch_task],
                            isaaclab_urdf_path,
                            isaaclab_urdf_mesh_path,
                            isaaclab_urdf_code_path,
                            isaaclab_mirror_urdf_path,
                            isaaclab_mirror_urdf_mesh_path,
                            isaaclab_mirror_urdf_code_path,
                            isaaclab_env_code_path,
                            isaaclab_test_result_path,
                            experiment_name=experiment_save_path,
                            evaluation_state_path=child_state_path,
                            individual_id=child["child_id"],
                            max_iterations=stage_max_iterations,
                            slot_id=slot_id,
                            curriculum_stage=stage_name,
                            all_evaluation_tasks=task_names,
                            inherited_checkpoint_path=inherited_checkpoint,
                            parent_individual_id=parent.get('id'),
                        )
                    except Exception as error:  # noqa: BLE001
                        evaluation_error = f"{type(error).__name__}: {error}"
                        current_score = float("-inf")
                        failed_state = _load_json(child_state_path) or {}
                        failed_state.update(
                            {
                                "status": "failed",
                                "current_task": failed_state.get("current_task"),
                                "error": evaluation_error,
                            }
                        )
                        _atomic_write_json(child_state_path, failed_state)
                        print(
                            f"[WARN] Slot {slot_id} failed child {child['child_id']} during {stage_name}: "
                            f"{evaluation_error}. Marking the child as failed and continuing.",
                            flush=True,
                        )

                    with state_lock:
                        if batch_task != final_task_name:
                            continue
                        completed_state = _load_json(child_state_path) or {}
                        completed_scores = dict(completed_state.get("task_scores", {}))
                        stage_complete = (
                            completed_state.get("status") == "completed"
                            and set(task_names).issubset(completed_scores)
                            and all(
                                math.isfinite(float(completed_scores[task_name]))
                                for task_name in task_names
                            )
                        )
                        score_value = (
                            current_score
                            if stage_complete and math.isfinite(current_score)
                            else float("-inf")
                        )
                        metadata_updates = {
                            f"{stage_name}_score": score_value,
                            f"{stage_name}_max_iterations": stage_max_iterations,
                            f"{stage_name}_complete": stage_complete,
                        }
                        if stage_name == "stage2":
                            metadata_updates["stage2_verified"] = stage_complete
                            if stage_complete:
                                metadata_updates["selected_curriculum_stage"] = "stage2"
                        if not stage_complete:
                            metadata_updates[f"{stage_name}_error"] = (
                                completed_state.get("error")
                                or evaluation_error
                                or "not_all_tasks_completed"
                            )
                        metadata_updates[f"{stage_name}_task_scores"] = dict(
                            completed_scores
                        )
                        metadata_updates[f"{stage_name}_run_names"] = {
                            key: value
                            for key, value in completed_state.get("run_names", {}).items()
                            if key.startswith(f"{stage_name}:")
                        }
                        metadata_updates['policy_initialization'] = dict(
                            completed_state.get('policy_initialization', {}))
                        stage_results[child["child_id"]] = score_value
                        if not hand_lineage.has_individual_id(child["child_id"]):
                            merged_metadata = dict(child["metadata"])
                            merged_metadata.update(metadata_updates)
                            hand_lineage.add_individual(
                                child["generation"],
                                child["individual"],
                                child["urdf_info"],
                                score_value,
                                child["child_id"],
                                metadata=merged_metadata,
                            )
                        else:
                            hand_lineage.update_individual_by_id(
                                child["child_id"],
                                task_score=score_value,
                                metadata_updates=metadata_updates,
                            )
                        hand_lineage.save_to_file(experiment_json_path)

                        pending_by_id.pop(child["child_id"], None)
                        remaining = list(pending_by_id.values())
                        _persist_generation_state(
                            runtime_state_json_path,
                            current_generation,
                            remaining,
                            parallel_slots,
                        )

            for batch_task in task_names:
                slot_queues = [[] for _ in range(parallel_slots)]
                for child in stage_children:
                    slot_queues[child["slot_id"]].append(child)

                with ThreadPoolExecutor(max_workers=parallel_slots) as executor:
                    futures = [
                        executor.submit(_run_slot_queue, slot_id, slot_children, batch_task)
                        for slot_id, slot_children in enumerate(slot_queues)
                        if slot_children
                    ]
                    for future in as_completed(futures):
                        future.result()

            return stage_results

        # A one-stage experiment can train directly against either the easy
        # stage1 or strict stage2 task definition without a restart boundary.
        stage1_results = _evaluate_stage(pending_children, stage1_max_iterations, single_stage_name)

        if stage2_enabled:
            if restart_workers_at_boundaries:
                close_evaluation_workers()
            ranked_children = sorted(
                pending_children,
                key=lambda child: stage1_results.get(child["child_id"], float("-inf")),
                reverse=True,
            )
            verified_parent_count = sum(
                bool(hand_lineage.lineage[(current_generation, parent_number)].get("metadata", {}).get("stage2_verified"))
                for parent_number in surviving_individuals
            )
            population_shortfall = (
                max(0, max_population - verified_parent_count)
                if retain_parent_elites
                else 0
            )
            top_k = min(
                len(ranked_children),
                max(
                    1,
                    math.ceil(len(ranked_children) * stage2_top_fraction),
                    population_shortfall,
                ),
            )
            print(
                f"[INFO] Stage2 selection: fraction_target={math.ceil(len(ranked_children) * stage2_top_fraction)} "
                f"verified_parent_elites={verified_parent_count} population_shortfall={population_shortfall} "
                f"selected_children={top_k}",
                flush=True,
            )
            stage2_children = [
                child
                for child in ranked_children[:top_k]
                if math.isfinite(stage1_results.get(child["child_id"], float("-inf")))
            ]
            stage2_results = {}
            if stage2_children:
                stage2_results = _evaluate_stage(
                    _assign_slots(stage2_children, parallel_slots), stage2_max_iterations, "stage2"
                )
            stage2_ids = {
                child_id for child_id, score in stage2_results.items() if math.isfinite(score)
            }
            for child in pending_children:
                if child["child_id"] in stage2_ids:
                    continue
                hand_lineage.update_individual_by_id(
                    child["child_id"],
                    task_score=float("-inf"),
                    metadata_updates={
                        "stage2_verified": False,
                        "selection_exclusion": "stage2_not_completed",
                    },
                    tag="eliminated",
                )
        elif restart_workers_at_boundaries:
            close_evaluation_workers()

    if retain_parent_elites:
        # With one child per parent, retain a scored parent as an unevaluated
        # candidate so selection remains 8 parents + 8 children -> top 8.
        # This halves new RL training without changing the population size.
        for parent_number in surviving_individuals:
            parent = hand_lineage.lineage[(current_generation, parent_number)]
            if stage2_enabled and not parent.get("metadata", {}).get("stage2_verified", False):
                continue
            elite_id = _make_elite_id(experiment_save_path, current_generation + 1, parent["id"])
            if hand_lineage.has_individual_id(elite_id):
                continue
            elite_metadata = dict(parent.get("metadata", {}))
            elite_metadata.update(
                {
                    "elite_copy": True,
                    "elite_from_generation": current_generation,
                    "elite_from_id": parent["id"],
                }
            )
            hand_lineage.add_individual(
                current_generation,
                parent_number,
                copy.deepcopy(parent["urdf_info"]),
                parent["task_score"],
                elite_id,
                metadata=elite_metadata,
            )
    hand_lineage.evaluate_and_eliminate_individuals_in_generation(current_generation + 1, max_population)
    if not hand_lineage.get_surviving_individuals_in_generation(current_generation + 1):
        hand_lineage.save_to_file(experiment_json_path)
        raise RuntimeError(
            f"Generation {current_generation + 1} has no surviving individuals after selection; "
            "task failures must be resolved before evolution can continue."
        )
    _cleanup_eliminated_children(
        experiment_save_path,
        current_generation + 1,
        hand_lineage,
        sorted(evaluation_taks),
    )
    hand_lineage.save_to_file(experiment_json_path)
    if restart_workers_at_boundaries:
        close_evaluation_workers()
    runtime_state = _save_runtime_state(
        runtime_state_json_path,
        _build_runtime_state(current_generation + 1, phase="ready", parallel_slots=parallel_slots),
    )
