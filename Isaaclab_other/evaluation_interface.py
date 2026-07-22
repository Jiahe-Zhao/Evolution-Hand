"""进化程序与 IsaacLab 仿真引擎的接口。"""

from code_to_urdf import generate_urdf_from_dict
from isaaclab_tool import (
    calculate_observation_number,
    parse_urdf_and_generate_articulation_cfg,
    task_generate_env_cfg,
)
import glob
import json
import os
import re
import signal
import subprocess
import time

from mirror_agent import create_mirror_hand
from read_results import (
    check_finished_folder_exists_in_run_dir,
    get_reward_from_run_dir,
)


HOME_DIR = os.path.expanduser("~")
EVOLUTION_ROOT = os.environ.get("EVOLUTION_ROOT", os.path.join(HOME_DIR, "Evolution_PC"))
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", os.path.join(HOME_DIR, "IsaacLab"))
ISAACLAB_TRAIN_SCRIPT = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "evolution_tasks", "train_interface.py"
)
ISAACLAB_ENV_PREFIX = os.environ.get("CONDA_PREFIX", os.path.join(HOME_DIR, "envs", "isaaclab"))
EVOLUTION_LOG_ROOT = os.environ.get(
    "EVOLUTION_LOG_ROOT",
    os.path.join(EVOLUTION_ROOT, "evolution_tasks", "logs"),
)
ISAAC_SIM_SETUP = os.environ.get(
    "ISAAC_SIM_SETUP",
    os.path.join(HOME_DIR, "isaac-sim-sandbox", "isaac-sim", "setup_conda_env.sh"),
)
ISAACLAB_NUM_ENVS = int(os.environ.get("ISAACLAB_NUM_ENVS", "256"))
ISAACLAB_MAX_ITERATIONS = os.environ.get("ISAACLAB_MAX_ITERATIONS")
ISAACLAB_CHECKPOINT_INTERVAL = int(os.environ.get("EVOLUTION_CHECKPOINT_INTERVAL", "20"))
ISAACLAB_STALL_TIMEOUT_SECONDS = max(60, int(os.environ.get("EVOLUTION_STALL_TIMEOUT_SECONDS", "900")))
ISAACLAB_TERMINATE_TIMEOUT_SECONDS = max(5, int(os.environ.get("EVOLUTION_TERM_TIMEOUT_SECONDS", "30")))
ISAACLAB_MAX_RESTARTS = max(0, int(os.environ.get("EVOLUTION_MAX_RESTARTS", "3")))
ISAACLAB_POLL_INTERVAL_SECONDS = max(5, int(os.environ.get("EVOLUTION_PROCESS_POLL_SECONDS", "15")))
RL_LOG_GROUP = "evolution_task"
STONEGRIND_TASK_NAME = "Isaac-EvolutionHand-StoneGrind-v0"
PARALLEL_SLOT_ROOT = os.path.join(EVOLUTION_ROOT, "parallel_eval_slots")

TASK_ENV_CFG_FILES = {
    "Isaac-EvolutionHand-StoneGrind-v0": os.path.join(
        ISAACLAB_ROOT,
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "evolution_tasks",
        "task_stone",
        "evolution_stone_grind_env_cfg.py",
    ),
    "Isaac-EvolutionHand-Grasp-v0": os.path.join(
        ISAACLAB_ROOT,
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "evolution_tasks",
        "task_grasp",
        "evolution_grasp_env_cfg.py",
    ),
    "Isaac-EvolutionHand-Strike-v0": os.path.join(
        ISAACLAB_ROOT,
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "evolution_tasks",
        "task_strike",
        "evolution_strike_env_cfg.py",
    ),
    "Isaac-EvolutionHand-Manipulation-v0": os.path.join(
        ISAACLAB_ROOT,
        "source",
        "isaaclab_tasks",
        "isaaclab_tasks",
        "evolution_tasks",
        "task_manipulation",
        "evolution_manipulation_env_cfg.py",
    ),
}

TASK_SLOT_RELATIVE_PATHS = {
    "Isaac-EvolutionHand-StoneGrind-v0": ("task_stone", "evolution_stone_grind_env_cfg.py"),
    "Isaac-EvolutionHand-Grasp-v0": ("task_grasp", "evolution_grasp_env_cfg.py"),
    "Isaac-EvolutionHand-Strike-v0": ("task_strike", "evolution_strike_env_cfg.py"),
    "Isaac-EvolutionHand-Manipulation-v0": ("task_manipulation", "evolution_manipulation_env_cfg.py"),
}


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def _slot_root(slot_id):
    return os.path.join(PARALLEL_SLOT_ROOT, f"slot_{slot_id}")


def _slot_override_root(slot_id):
    return os.path.join(_slot_root(slot_id), "python_overrides")


def _slot_package_path(slot_id, *parts):
    return os.path.join(_slot_override_root(slot_id), "isaaclab_tasks", "evolution_tasks", *parts)


def _slot_agent_paths(slot_id):
    agent_root = os.path.join(_slot_root(slot_id), "agent_for_isaaclab")
    mirror_root = os.path.join(_slot_root(slot_id), "agent_for_isaaclab_mirror")
    return {
        "override_root": _slot_override_root(slot_id),
        "right_urdf": os.path.join(agent_root, "urdf", "current_agent.urdf"),
        "right_mesh": os.path.join(agent_root, "mesh"),
        "right_cfg": _slot_package_path(slot_id, "current_right_hand", "current_right_hand_cfg.py"),
        "left_urdf": os.path.join(mirror_root, "urdf", "current_agent.urdf"),
        "left_mesh": os.path.join(mirror_root, "mesh"),
        "left_cfg": _slot_package_path(slot_id, "current_left_hand", "current_left_hand_cfg.py"),
    }


def _slot_task_cfg_path(slot_id, task_name):
    task_relative_parts = TASK_SLOT_RELATIVE_PATHS.get(task_name)
    if task_relative_parts is None:
        raise ValueError(f"Unsupported slot task: {task_name}")
    return _slot_package_path(slot_id, *task_relative_parts)


def _ensure_slot_override(slot_id):
    override_root = _slot_override_root(slot_id)
    sitecustomize_path = os.path.join(override_root, "sitecustomize.py")
    sitecustomize = f"""import importlib
import os

_OVERRIDE_ROOT = {override_root!r}
_ISAACLAB_TASKS_ROOT = os.path.join(_OVERRIDE_ROOT, "isaaclab_tasks")
_EVOLUTION_TASKS_ROOT = os.path.join(_ISAACLAB_TASKS_ROOT, "evolution_tasks")
_SUBPKG_PATHS = {{
    "isaaclab_tasks": _ISAACLAB_TASKS_ROOT,
    "isaaclab_tasks.evolution_tasks": _EVOLUTION_TASKS_ROOT,
    "isaaclab_tasks.evolution_tasks.current_right_hand": os.path.join(_EVOLUTION_TASKS_ROOT, "current_right_hand"),
    "isaaclab_tasks.evolution_tasks.current_left_hand": os.path.join(_EVOLUTION_TASKS_ROOT, "current_left_hand"),
    "isaaclab_tasks.evolution_tasks.task_grasp": os.path.join(_EVOLUTION_TASKS_ROOT, "task_grasp"),
    "isaaclab_tasks.evolution_tasks.task_strike": os.path.join(_EVOLUTION_TASKS_ROOT, "task_strike"),
    "isaaclab_tasks.evolution_tasks.task_manipulation": os.path.join(_EVOLUTION_TASKS_ROOT, "task_manipulation"),
    "isaaclab_tasks.evolution_tasks.task_stone": os.path.join(_EVOLUTION_TASKS_ROOT, "task_stone"),
}}

for _module_name, _path in _SUBPKG_PATHS.items():
    if not os.path.isdir(_path):
        continue
    _module = importlib.import_module(_module_name)
    _module_path = list(getattr(_module, "__path__", []))
    if _path not in _module_path:
        _module.__path__.insert(0, _path)
"""
    _write_text(sitecustomize_path, sitecustomize)
    for relative_dir in (
        ("isaaclab_tasks", "evolution_tasks", "current_right_hand"),
        ("isaaclab_tasks", "evolution_tasks", "current_left_hand"),
        ("isaaclab_tasks", "evolution_tasks", "task_grasp"),
        ("isaaclab_tasks", "evolution_tasks", "task_strike"),
        ("isaaclab_tasks", "evolution_tasks", "task_manipulation"),
        ("isaaclab_tasks", "evolution_tasks", "task_stone"),
    ):
        os.makedirs(os.path.join(override_root, *relative_dir), exist_ok=True)


def _ensure_slot_agent_dirs(slot_paths):
    os.makedirs(os.path.dirname(slot_paths["right_urdf"]), exist_ok=True)
    os.makedirs(slot_paths["right_mesh"], exist_ok=True)
    os.makedirs(os.path.dirname(slot_paths["left_urdf"]), exist_ok=True)
    os.makedirs(slot_paths["left_mesh"], exist_ok=True)


def _task_slug(task_name):
    return (
        task_name.replace("Isaac-", "")
        .replace("-v0", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _make_task_run_name(experiment_name, individual_id, task_name):
    return f"{experiment_name}_{individual_id[:8]}_{_task_slug(task_name)}"


def _task_run_dir(run_name):
    return os.path.join(EVOLUTION_LOG_ROOT, RL_LOG_GROUP, run_name)


def _num_envs_for_task(task_name):
    if task_name != STONEGRIND_TASK_NAME:
        return _split_num_envs_for_parallel(ISAACLAB_NUM_ENVS)

    raw_override = os.environ.get("EVOLUTION_STONEGRIND_NUM_ENVS")
    if raw_override not in (None, ""):
        return _split_num_envs_for_parallel(int(raw_override))

    # StoneGrind is much heavier than the other tasks because it instantiates
    # two hands, two rigid bodies and contact sensing per environment.
    return _split_num_envs_for_parallel(min(ISAACLAB_NUM_ENVS, 64))


def _parallel_slots():
    raw_value = os.environ.get("EVOLUTION_PARALLEL_SLOTS")
    if raw_value in (None, ""):
        return 1
    return max(1, int(raw_value))


def _split_num_envs_for_parallel(base_num_envs):
    split_enabled = os.environ.get("EVOLUTION_PARALLEL_SPLIT_ENVS", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    slots = _parallel_slots()
    if not split_enabled or slots <= 1:
        return base_num_envs
    return max(1, base_num_envs // slots)


def _find_latest_checkpoint(run_dir):
    nn_dir = os.path.join(run_dir, "nn")
    if not os.path.isdir(nn_dir):
        return None
    checkpoints = glob.glob(os.path.join(nn_dir, "*.pth"))
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def _checkpoint_epoch(checkpoint_path):
    if not checkpoint_path:
        return None
    match = re.search(r"_ep_(\d+)_", os.path.basename(checkpoint_path))
    if not match:
        return None
    return int(match.group(1))


def _latest_checkpoint_state(run_dir):
    checkpoint_path = _find_latest_checkpoint(run_dir)
    if not checkpoint_path:
        return None, None, None
    try:
        checkpoint_mtime = os.path.getmtime(checkpoint_path)
    except OSError:
        checkpoint_mtime = None
    return checkpoint_path, _checkpoint_epoch(checkpoint_path), checkpoint_mtime


def _terminate_process_group(process):
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + ISAACLAB_TERMINATE_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return

    deadline = time.time() + 5
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)


def evaluation(
    urdf_dic,
    evaluation_tasks,
    isaaclab_urdf_path,
    isaaclab_urdf_mesh_path,
    isaaclab_urdf_code_path,
    isaaclab_mirror_urdf_path,
    isaaclab_mirror_urdf_mesh_path,
    isaaclab_mirror_urdf_code_path,
    isaaclab_env_code_path,
    isaaclab_test_result_path,
    experiment_name="evolution_multitask",
    evaluation_state_path=None,
    individual_id=None,
    max_iterations=None,
    slot_id=0,
):
    ordered_tasks = sorted(evaluation_tasks)
    if not ordered_tasks:
        raise ValueError("evaluation_tasks is empty")
    if not individual_id:
        raise ValueError("individual_id is required for resumable evaluation")

    slot_paths = _slot_agent_paths(slot_id)
    _ensure_slot_override(slot_id)
    _ensure_slot_agent_dirs(slot_paths)

    isaaclab_urdf_path = slot_paths["right_urdf"]
    isaaclab_urdf_mesh_path = slot_paths["right_mesh"]
    isaaclab_urdf_code_path = slot_paths["right_cfg"]
    isaaclab_mirror_urdf_path = slot_paths["left_urdf"]
    isaaclab_mirror_urdf_mesh_path = slot_paths["left_mesh"]
    isaaclab_mirror_urdf_code_path = slot_paths["left_cfg"]

    delete_urdf_and_stl_files(isaaclab_urdf_path, isaaclab_urdf_mesh_path)
    generate_urdf_from_dict(urdf_dic, output_dir=isaaclab_urdf_mesh_path, output_urdf=isaaclab_urdf_path)
    parse_urdf_and_generate_articulation_cfg(isaaclab_urdf_path, isaaclab_urdf_path, isaaclab_urdf_code_path)

    mirror_hand = create_mirror_hand(urdf_dic, "mirror_agent_hand")
    delete_urdf_and_stl_files(isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_mesh_path)
    generate_urdf_from_dict(
        mirror_hand,
        output_dir=isaaclab_mirror_urdf_mesh_path,
        output_urdf=isaaclab_mirror_urdf_path,
    )
    parse_urdf_and_generate_articulation_cfg(
        isaaclab_mirror_urdf_path,
        isaaclab_mirror_urdf_path,
        isaaclab_mirror_urdf_code_path,
    )

    observation_number = calculate_observation_number(isaaclab_urdf_path)

    state = None
    if evaluation_state_path:
        state = _load_json(evaluation_state_path)
    if not state or state.get("individual_id") != individual_id:
        state = {
            "version": 1,
            "status": "running",
            "individual_id": individual_id,
            "experiment_name": experiment_name,
            "ordered_tasks": ordered_tasks,
            "task_scores": {},
            "run_names": {},
            "current_task": None,
            "max_iterations": max_iterations,
        }
        if evaluation_state_path:
            _atomic_write_json(evaluation_state_path, state)

    stored_max_iterations = state.get("max_iterations")
    rerun_completed = (
        state.get("status") == "completed"
        and max_iterations is not None
        and stored_max_iterations is not None
        and max_iterations > stored_max_iterations
    )
    if state.get("status") == "completed" and not rerun_completed:
        average_score = sum(state["task_scores"].values()) / len(state["task_scores"])
        print(f"task_scores:{state['task_scores']}")
        print(f"average_score:{average_score}")
        return average_score
    if rerun_completed:
        state["status"] = "running"
        state["task_scores"] = {}
        state["current_task"] = None
        state["max_iterations"] = max_iterations
        if evaluation_state_path:
            _atomic_write_json(evaluation_state_path, state)

    task_scores = dict(state.get("task_scores", {}))
    run_names = dict(state.get("run_names", {}))
    effective_max_iterations = max_iterations if max_iterations is not None else state.get("max_iterations")

    for current_task in ordered_tasks:
        if current_task in task_scores:
            continue

        task_env_cfg_path = _slot_task_cfg_path(slot_id, current_task)
        task_generate_env_cfg(
            current_task,
            isaaclab_urdf_path,
            isaaclab_mirror_urdf_path,
            observation_number,
            task_env_cfg_path,
        )

        run_name = run_names.get(current_task) or _make_task_run_name(experiment_name, individual_id, current_task)
        run_names[current_task] = run_name
        run_dir = _task_run_dir(run_name)

        state["run_names"] = run_names
        state["current_task"] = current_task
        state["max_iterations"] = effective_max_iterations
        if evaluation_state_path:
            _atomic_write_json(evaluation_state_path, state)

        if check_finished_folder_exists_in_run_dir(run_dir):
            score = get_reward_from_run_dir(run_dir)
        else:
            checkpoint_path = _find_latest_checkpoint(run_dir)
            task_num_envs = _num_envs_for_task(current_task)
            print(f"[INFO] Launching {current_task} with num_envs={task_num_envs}")
            score = run_isaaclab_simulation(
                current_task,
                isaaclab_test_result_path,
                num_envs=task_num_envs,
                run_name=run_name,
                checkpoint_path=checkpoint_path,
                max_iterations=effective_max_iterations,
                slot_id=slot_id,
                python_override_root=slot_paths["override_root"],
            )

        task_scores[current_task] = score
        state["task_scores"] = task_scores
        state["current_task"] = None
        if evaluation_state_path:
            _atomic_write_json(evaluation_state_path, state)

    average_score = sum(task_scores.values()) / len(task_scores)
    state["status"] = "completed"
    state["task_scores"] = task_scores
    if evaluation_state_path:
        _atomic_write_json(evaluation_state_path, state)
    print(f"task_scores:{task_scores}")
    print(f"average_score:{average_score}")
    return average_score


def delete_urdf_and_stl_files(directory_urdf, directory_mesh):
    """Delete generated URDF and STL files before regenerating the hand."""
    if os.path.isdir(directory_urdf):
        urdf_files = glob.glob(os.path.join(directory_urdf, "*.urdf"))
    else:
        urdf_files = [directory_urdf] if os.path.exists(directory_urdf) else []
    stl_files = glob.glob(os.path.join(directory_mesh, "*.stl"))
    for file_path in urdf_files + stl_files:
        try:
            os.remove(file_path)
        except Exception as error:  # noqa: BLE001
            print(f"删除文件 {file_path} 时出错: {error}")


def check_previous_simulation():
    result = subprocess.run("ps aux | grep isaaclab", shell=True, capture_output=True, text=True)
    return result.stdout


def run_isaaclab_simulation(
    task_name,
    isaaclab_test_result_path,
    num_envs=ISAACLAB_NUM_ENVS,
    run_name=None,
    checkpoint_path=None,
    max_iterations=None,
    slot_id=0,
    python_override_root=None,
):
    effective_max_iterations = max_iterations
    if effective_max_iterations is None and ISAACLAB_MAX_ITERATIONS:
        effective_max_iterations = int(ISAACLAB_MAX_ITERATIONS)

    run_dir = _task_run_dir(run_name) if run_name else None
    next_checkpoint_path = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None
    restart_count = 0

    while True:
        command = [
            os.path.join(ISAACLAB_ROOT, "isaaclab.sh"),
            "-p",
            ISAACLAB_TRAIN_SCRIPT,
            "--num_envs",
            str(num_envs),
            "--task",
            str(task_name),
            "--device",
            "cuda:0",
            "--headless",
        ]
        if run_name:
            command.extend(["--run_name", run_name])
        if next_checkpoint_path and os.path.exists(next_checkpoint_path):
            command.extend(["--checkpoint", next_checkpoint_path])
        if effective_max_iterations is not None:
            command.extend(["--max_iterations", str(effective_max_iterations)])
        if ISAACLAB_CHECKPOINT_INTERVAL > 0:
            command.extend(["--checkpoint_interval", str(ISAACLAB_CHECKPOINT_INTERVAL)])

        shell_cmd = f"""
        export TERM=xterm
        export ACCEPT_EULA=Y
        unset CUDA_VISIBLE_DEVICES
        export CONDA_PREFIX="{ISAACLAB_ENV_PREFIX}"
        export PATH="{ISAACLAB_ENV_PREFIX}/bin:$PATH"
        export EVOLUTION_LOG_ROOT="{EVOLUTION_LOG_ROOT}"
        export EVOLUTION_PARALLEL_SLOT="{slot_id}"
        export PYTHONPATH="{python_override_root or ''}:$PYTHONPATH"
        set +u
        source "{ISAAC_SIM_SETUP}"
        set -u
        cd "{ISAACLAB_ROOT}"
        {' '.join(command)}
        """

        print(f"[INFO] Starting IsaacLab run_name={run_name} checkpoint={next_checkpoint_path} max_iterations={effective_max_iterations}")
        process = subprocess.Popen(
            ["bash", "-c", shell_cmd],
            start_new_session=True,
        )

        last_progress_time = time.time()
        _, _, last_checkpoint_mtime = _latest_checkpoint_state(run_dir) if run_dir else (None, None, None)
        stalled = False

        while True:
            return_code = process.poll()
            latest_checkpoint_path, latest_epoch, latest_checkpoint_mtime = (
                _latest_checkpoint_state(run_dir) if run_dir else (None, None, None)
            )
            if latest_checkpoint_mtime is not None and (
                last_checkpoint_mtime is None or latest_checkpoint_mtime > last_checkpoint_mtime
            ):
                last_checkpoint_mtime = latest_checkpoint_mtime
                last_progress_time = time.time()

            reward = get_reward_from_run_dir(run_dir) if run_dir else None
            if return_code is not None:
                break

            if (
                reward is not None
                and effective_max_iterations is not None
                and latest_epoch is not None
                and latest_epoch >= effective_max_iterations
                and time.time() - last_progress_time >= ISAACLAB_POLL_INTERVAL_SECONDS * 2
            ):
                print(
                    f"[WARN] IsaacLab reached checkpoint epoch {latest_epoch} with reward {reward}, but process is still alive. Forcing shutdown."
                )
                _terminate_process_group(process)
                break

            if time.time() - last_progress_time >= ISAACLAB_STALL_TIMEOUT_SECONDS:
                stalled = True
                print(
                    f"[WARN] IsaacLab appears stalled for {int(time.time() - last_progress_time)}s; latest checkpoint={latest_checkpoint_path}, epoch={latest_epoch}. Restarting from latest checkpoint."
                )
                _terminate_process_group(process)
                break

            time.sleep(ISAACLAB_POLL_INTERVAL_SECONDS)

        print("Simulation finished.")

        if run_dir and check_finished_folder_exists_in_run_dir(run_dir):
            score = get_reward_from_run_dir(run_dir)
            return score

        reward = get_reward_from_run_dir(run_dir) if run_dir else None
        if reward is not None:
            print("Finished marker missing, but reward checkpoint exists. Using checkpoint reward directly.")
            return reward

        next_checkpoint_path = _find_latest_checkpoint(run_dir) if run_dir else next_checkpoint_path
        latest_epoch = _checkpoint_epoch(next_checkpoint_path)
        if (
            effective_max_iterations is not None
            and latest_epoch is not None
            and latest_epoch >= effective_max_iterations
        ):
            reward = get_reward_from_run_dir(run_dir) if run_dir else None
            if reward is not None:
                print("Latest checkpoint reached max_iterations. Using checkpoint reward directly.")
                return reward

        should_retry = stalled or process.returncode not in (0, None)
        if should_retry and restart_count < ISAACLAB_MAX_RESTARTS:
            restart_count += 1
            print(
                f"[WARN] Restarting IsaacLab run {run_name} ({restart_count}/{ISAACLAB_MAX_RESTARTS}) from checkpoint {next_checkpoint_path}."
            )
            continue

        time.sleep(10)
        print("simulation running")
        return float("-4000")

    shell_cmd = f"""
    export TERM=xterm
    export ACCEPT_EULA=Y
    unset CUDA_VISIBLE_DEVICES
    export CONDA_PREFIX="{ISAACLAB_ENV_PREFIX}"
    export PATH="{ISAACLAB_ENV_PREFIX}/bin:$PATH"
    export EVOLUTION_LOG_ROOT="{EVOLUTION_LOG_ROOT}"
    export EVOLUTION_PARALLEL_SLOT="{slot_id}"
    export PYTHONPATH="{python_override_root or ''}:$PYTHONPATH"
    set +u
    source "{ISAAC_SIM_SETUP}"
    set -u
    cd "{ISAACLAB_ROOT}"
    {' '.join(command)}
    """

    process = subprocess.Popen(
        ["bash", "-c", shell_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    print(f"stdout:{stdout}")
    print(f"stderr:{stderr}")
    print("Simulation finished.")

    time.sleep(3)
    if not run_name:
        time.sleep(10)
        print("simulation running")
        return float("-4000")

    run_dir = _task_run_dir(run_name)
    if check_finished_folder_exists_in_run_dir(run_dir):
        score = get_reward_from_run_dir(run_dir)
        return score
    reward = get_reward_from_run_dir(run_dir)
    if reward is not None:
        print("Finished marker missing, but reward checkpoint exists. Using checkpoint reward directly.")
        return reward
    time.sleep(10)
    print("simulation running")
    return float("-4000")
