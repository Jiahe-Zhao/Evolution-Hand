"""主要的进化入口，支持外层与内层断点续跑，以及同代个体并行评估。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import random
import shutil
import threading
import uuid

import numpy as np

from class_population import Lineage
from evaluation_interface import evaluation
from variation import choose_target, seed_initial_population, variation


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
    else:
        raise ValueError(f"Unsupported check_point: {check_point}")

    initial_population = seed_initial_population(
        initial_agent_hand,
        population_size=initial_population_size,
        include_base=True,
        max_attempts=initial_population_attempts,
        standard_variation=initial_population_variation,
        standard_length=initial_population_length,
    )
    for idx, urdf in enumerate(initial_population):
        new_id = uuid.uuid4().hex
        urdf["evolution_id"] = new_id
        lineage.add_individual(-1, idx, urdf, 0, new_id, metadata={"seed_stage": "initial_population"})
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
    variation_probabilities,
):
    if runtime_state["phase"] == "evaluating" and runtime_state.get("pending_children"):
        return [
            child
            for child in runtime_state["pending_children"]
            if child["generation"] == current_generation and not hand_lineage.has_individual_id(child["child_id"])
        ]

    pending_lookup = {}
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
        for trial in range(start_trial, max_variation):
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
            if child is None:
                continue
            if hand_lineage.has_individual_id(child["child_id"]):
                continue
            children.append(child)

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


def _make_task_run_name(experiment_name, individual_id, task_name):
    return f"{experiment_name}_{individual_id[:8]}_{_task_slug(task_name)}"


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

        for task_name in ordered_tasks:
            run_name = run_names.get(task_name) or _make_task_run_name(experiment_name, child_id, task_name)
            if _remove_path(os.path.join(EVOLUTION_LOG_ROOT, run_name)):
                removed_run_dirs += 1

        if _remove_path(child_state_path):
            removed_state_files += 1

    if removed_run_dirs or removed_state_files:
        print(
            f"[INFO] Cleaned eliminated children for generation {generation}: "
            f"run_dirs={removed_run_dirs}, state_files={removed_state_files}"
        )


# 基本配置
# Treat EVOLUTION_MAX_GENERATION as the total number of parent generations,
# starting from generation 0.
max_generation = _env_int("EVOLUTION_MAX_GENERATION", 1000)
max_population = _env_int("EVOLUTION_MAX_POPULATION", 1000)
max_variation = _env_int("EVOLUTION_MAX_VARIATION", 10)
parallel_slots = max(1, _env_int("EVOLUTION_PARALLEL_SLOTS", 1))
initial_population_size = _env_int("EVOLUTION_INITIAL_POPULATION_SIZE", 8)
initial_population_attempts = _env_int("EVOLUTION_INITIAL_POPULATION_ATTEMPTS", 200)
initial_population_variation = _env_float("EVOLUTION_INITIAL_POPULATION_VARIATION", 0.05)
initial_population_length = _env_float("EVOLUTION_INITIAL_POPULATION_LENGTH", 0.02)
variation_probabilities = {
    "change_link_length": 1.0 / 30.0,
    "change_link_radius": 1.0 / 30.0,
    "remove_link": 1.0 / 30.0,
    "add_link": 1.0 / 30.0,
    "change_joint_origin_translation": 1.0 / 30.0,
    "change_joint_origin_rpy": 1.0 / 30.0,
    "change_thumb_length": 0.4,
    "change_palm_curvature": 0.4,
}
experiment_save_path = os.environ.get("EVOLUTION_EXPERIMENT_NAME", "exp_20260709_multitask_1")
evaluation_tasks_env = os.environ.get("EVOLUTION_TASKS")
evaluation_taks = (
    {task.strip() for task in evaluation_tasks_env.split(",") if task.strip()}
    if evaluation_tasks_env
    else {
        "Isaac-EvolutionHand-StoneGrind-v0",
        "Isaac-EvolutionHand-Grasp-v0",
        "Isaac-EvolutionHand-Strike-v0",
        "Isaac-EvolutionHand-Manipulation-v0",
    }
)
check_point = os.environ.get("EVOLUTION_INITIAL_AGENT", "human")
force_new_lineage = _env_flag("EVOLUTION_FORCE_NEW_LINEAGE", False)
inner_max_iterations_env = os.environ.get("ISAACLAB_MAX_ITERATIONS")
inner_max_iterations = int(inner_max_iterations_env) if inner_max_iterations_env else None
stage1_max_iterations = _env_int(
    "EVOLUTION_STAGE1_MAX_ITERATIONS",
    inner_max_iterations if inner_max_iterations is not None else 200,
)
stage2_max_iterations = _env_int("EVOLUTION_STAGE2_MAX_ITERATIONS", 500)
stage2_top_fraction = min(1.0, max(0.0, _env_float("EVOLUTION_STAGE2_TOP_FRACTION", 0.2)))
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
        runtime_state = _save_runtime_state(
            runtime_state_json_path,
            _build_runtime_state(current_generation + 1, phase="ready", parallel_slots=parallel_slots),
        )
        continue

    pending_children = _collect_pending_children(
        runtime_state,
        current_generation,
        surviving_individuals,
        hand_lineage,
        experiment_save_path,
        max_variation,
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

            def _run_slot_queue(slot_id, slot_children):
                for child in slot_children:
                    child_state_path = _evaluation_state_path_for_child(experiment_save_path, child["child_id"])
                    _migrate_legacy_evaluation_state(
                        legacy_evaluation_state_json_path,
                        child_state_path,
                        child["child_id"],
                    )
                    current_score = evaluation(
                        child["urdf_info"],
                        evaluation_taks,
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
                    )

                    score_value = current_score if math.isfinite(current_score) else float("-inf")
                    metadata_updates = {
                        f"{stage_name}_score": score_value,
                        f"{stage_name}_max_iterations": stage_max_iterations,
                    }

                    with state_lock:
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

            slot_queues = [[] for _ in range(parallel_slots)]
            for child in stage_children:
                slot_queues[child["slot_id"]].append(child)

            with ThreadPoolExecutor(max_workers=parallel_slots) as executor:
                futures = [
                    executor.submit(_run_slot_queue, slot_id, slot_children)
                    for slot_id, slot_children in enumerate(slot_queues)
                    if slot_children
                ]
                for future in as_completed(futures):
                    future.result()

            return stage_results

        stage1_results = _evaluate_stage(pending_children, stage1_max_iterations, "stage1")

        stage2_enabled = stage2_max_iterations > stage1_max_iterations and stage2_top_fraction > 0.0
        if stage2_enabled:
            ranked_children = sorted(
                pending_children,
                key=lambda child: stage1_results.get(child["child_id"], float("-inf")),
                reverse=True,
            )
            top_k = max(1, math.ceil(len(ranked_children) * stage2_top_fraction))
            stage2_children = [
                child
                for child in ranked_children[:top_k]
                if math.isfinite(stage1_results.get(child["child_id"], float("-inf")))
            ]
            if stage2_children:
                _evaluate_stage(_assign_slots(stage2_children, parallel_slots), stage2_max_iterations, "stage2")

    hand_lineage.evaluate_and_eliminate_individuals_in_generation(current_generation + 1, max_population)
    _cleanup_eliminated_children(
        experiment_save_path,
        current_generation + 1,
        hand_lineage,
        sorted(evaluation_taks),
    )
    hand_lineage.save_to_file(experiment_json_path)
    runtime_state = _save_runtime_state(
        runtime_state_json_path,
        _build_runtime_state(current_generation + 1, phase="ready", parallel_slots=parallel_slots),
    )
