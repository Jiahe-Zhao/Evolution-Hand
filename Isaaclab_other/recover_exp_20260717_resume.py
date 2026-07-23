import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXP = "exp_20260717_evolutionpc_grasponly_25g_stage2_tol3"
ROOT = Path("/home/zjh/Evolution_PC/Isaaclab_other")
LINEAGE_PATH = ROOT / f"{EXP}.json"
STATE_PATH = ROOT / f"{EXP}_runtime_state.json"
EVAL_DIR = ROOT / f"{EXP}_evaluation_states"
RUN_SCRIPT = ROOT / "run_local_evolution_4090.sh"
PARALLEL_SLOTS = 2
TARGET_GENERATION = 23
MAX_POPULATION = 16

os.environ.setdefault("EVOLUTION_ROOT", "/home/zjh/Evolution_PC")
os.environ.setdefault("ISAACLAB_ROOT", "/home/zjh/IsaacLab")
os.environ.setdefault("EVOLUTION_LOG_ROOT", "/home/zjh/Evolution_PC/evolution_tasks/logs")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("PRIVACY_CONSENT", "Y")
os.environ.setdefault("TERM", "xterm")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("CONDA_PREFIX", "/home/zjh/miniconda3/envs/evolution_isaaclab")
os.environ.setdefault("PATH", "/home/zjh/miniconda3/envs/evolution_isaaclab/bin:" + os.environ.get("PATH", ""))
os.environ.setdefault("ISAACLAB_NUM_ENVS", "128")
os.environ.setdefault("EVOLUTION_PARALLEL_SLOTS", str(PARALLEL_SLOTS))
os.environ.setdefault("EVOLUTION_STAGE1_MAX_ITERATIONS", "200")
os.environ.setdefault("EVOLUTION_STAGE2_MAX_ITERATIONS", "500")
os.environ.setdefault("EVOLUTION_STAGE2_TOP_FRACTION", "0.2")
os.environ.setdefault("EVOLUTION_STANDARD_VARIATION", "0.2")
os.environ.setdefault("EVOLUTION_STANDARD_LENGTH", "0.1")
os.environ.setdefault("EVOLUTION_CHECKPOINT_INTERVAL", "10")
os.environ.setdefault("EVOLUTION_TASKS", "Isaac-EvolutionHand-Grasp-v0")
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")
os.environ.setdefault("DISABLE_DEFAULT_GROUND_PLANE", "1")

from class_population import Lineage
from evaluation_interface import evaluation


def load_lineage():
    lineage = Lineage()
    lineage.load_from_file(str(LINEAGE_PATH))
    return lineage


def find_key_by_id(lineage, individual_id):
    for key, individual in lineage.lineage.items():
        if individual.get("id") == individual_id:
            return key, individual
    return None, None


def recover_one(lineage, state_path, slot_id):
    state = json.load(open(state_path))
    individual_id = state["individual_id"]
    key, individual = find_key_by_id(lineage, individual_id)
    if individual is None:
        raise RuntimeError(f"individual not found in lineage: {individual_id}")
    ordered_tasks = state.get("ordered_tasks") or ["Isaac-EvolutionHand-Grasp-v0"]
    max_iterations = state.get("max_iterations")
    score = evaluation(
        individual["urdf_info"],
        ordered_tasks,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        os.environ["EVOLUTION_LOG_ROOT"],
        experiment_name=EXP,
        evaluation_state_path=str(state_path),
        individual_id=individual_id,
        max_iterations=max_iterations,
        slot_id=slot_id,
    )
    lineage.update_individual_by_id(
        individual_id,
        task_score=score if math.isfinite(score) else float("-inf"),
        metadata_updates={
            "stage2_score": score if math.isfinite(score) else float("-inf"),
            "stage2_max_iterations": max_iterations,
        },
    )
    return individual_id, score, key


def main():
    pending = []
    for path in sorted(EVAL_DIR.glob("*.json")):
        state = json.load(open(path))
        if state.get("status") != "completed":
            pending.append(path)
    print(f"[RECOVER] unfinished evaluation states: {len(pending)}")

    lineage = load_lineage()

    if pending:
        with ThreadPoolExecutor(max_workers=min(PARALLEL_SLOTS, len(pending))) as ex:
            futures = {
                ex.submit(recover_one, lineage, path, idx % PARALLEL_SLOTS): path
                for idx, path in enumerate(pending)
            }
            for fut in as_completed(futures):
                individual_id, score, key = fut.result()
                print(f"[RECOVER] finished {individual_id} ({key}) score={score}")
                lineage.save_to_file(str(LINEAGE_PATH))

    lineage.evaluate_and_eliminate_individuals_in_generation(TARGET_GENERATION, MAX_POPULATION)
    lineage.save_to_file(str(LINEAGE_PATH))

    runtime_state = {
        "version": 2,
        "current_generation": TARGET_GENERATION,
        "current_individual": None,
        "trial": 0,
        "phase": "ready",
        "pending_child": None,
        "pending_children": [],
        "parallel_slots": PARALLEL_SLOTS,
    }
    with open(STATE_PATH, "w") as f:
        json.dump(runtime_state, f, indent=2)
    print(f"[RECOVER] runtime state reset to generation {TARGET_GENERATION} ready")

    print("[RECOVER] launching main evolution run...")
    subprocess.run(["bash", str(RUN_SCRIPT)], check=True)


if __name__ == "__main__":
    main()
