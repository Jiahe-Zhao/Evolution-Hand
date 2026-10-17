"""Opt-in mechanical-phenotype evaluation for the existing bone-genotype loop."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mechanical_evolution_gate import build_gated_phenotype, fingerprint


TASKS = {
    "Isaac-EvolutionHand-Grasp-v0": "grasp",
    "Isaac-EvolutionHand-BranchGrasp-v0": "branch",
    "Isaac-EvolutionHand-Forage-v0": "forage",
    "Isaac-EvolutionHand-Strike-v0": "strike",
}


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def _run(command, log_path):
    timeout = int(os.environ.get("EVOLUTION_MECHANICAL_TASK_TIMEOUT", "3600"))
    with Path(log_path).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Mechanical task exited {result.returncode}; inspect {log_path}")


def _reference(task):
    seeds=os.environ.get('EVOLUTION_MECHANICAL_BC_SEEDS')
    if seeds:
        manifest=json.loads(Path(seeds).read_text())
        if set(manifest['tasks']) != set(TASKS.values()):
            raise ValueError('Four verified BC seed checkpoints are required')
        entry=manifest['tasks'][task]
        source=Path(entry['checkpoint'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('BC seed checkpoint hash mismatch')
        return source
    root = Path(os.environ.get("EVOLUTION_MECHANICAL_REFERENCE_RESULTS",
        str(Path(__file__).with_name("mechanical_mapping_v4") / "ppo_integration")))
    report = json.loads((root / f"{task}4" / "mechanical_training.json").read_text())
    source = Path(report["source_checkpoint"])
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _legacy_urdf():
    path = Path(os.environ.get("EVOLUTION_MECHANICAL_LEGACY_URDF",
        str(Path(__file__).with_name("evaluation_results") / "preflight_grasp_projected_20260922"
            / "morphology" / "0_0" / "right" / "urdf" / "current_agent.urdf")))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _base_command(task, root, stage, checkpoint, output):
    return [os.environ.get("EVOLUTION_MECHANICAL_PYTHON", sys.executable),
            str(Path(__file__).with_name("evaluate_mechanical_task.py")),
            "--task", task, "--checkpoint", str(checkpoint),
            "--curriculum_stage", "stage2", "--legacy_source_urdf", str(_legacy_urdf()),
            "--mechanical_asset_dir", str(root / "assets"),
            "--lineage_json", str(root / "lineage.json"), "--individual_key", "candidate",
            "--output_dir", str(output), "--headless",
            *(["--mechanical_joint_actions"] if task in ("grasp","forage") else [])]


def _score(report):
    episode = report["episode_records"][0]
    if not episode["trace"]:
        raise ValueError("Mechanical evaluation has no physical trajectory")
    return sum(float(step["reward"]) for step in episode["trace"])


def evaluation(hand, requested_tasks, ordered_tasks, experiment_name, state_path,
               individual_id, max_iterations, stage, inherited_checkpoint_path,
               parent_individual_id=None):
    from evaluation_interface import _fitness_from_task_scores, _make_task_run_name

    unsupported = set(ordered_tasks) - set(TASKS)
    if unsupported:
        raise ValueError(f"Mechanical phenotype does not support tasks: {sorted(unsupported)}")
    if not state_path or not individual_id or max_iterations is None:
        raise ValueError("Mechanical evolution requires resumable state, individual ID and iteration limit")
    train_envs = int(os.environ.get("EVOLUTION_MECHANICAL_NUM_ENVS", "32"))
    if train_envs < 4:
        raise ValueError("Mechanical PPO requires at least four environments")
    experiment = Path(experiment_name).name
    output = Path(os.environ.get("EVOLUTION_MECHANICAL_OUTPUT_ROOT",
        str(Path(__file__).with_name("outputs") / "mechanical_evolution"))) / experiment
    output.mkdir(parents=True, exist_ok=True)
    with (output / (fingerprint(hand) + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root, gate = build_gated_phenotype(hand, output / "phenotypes")
    if not gate["passed"]:
        raise ValueError(f"Mechanical phenotype rejected: {gate.get('reason')}; {root / 'phenotype_gate.json'}")

    stage = (stage or "stage1").lower()
    state = json.loads(Path(state_path).read_text()) if Path(state_path).is_file() else {}
    if state.get("individual_id") != individual_id:
        state = {"version": 2, "status": "running", "individual_id": individual_id,
                 "experiment_name": experiment_name, "ordered_tasks": list(ordered_tasks),
                 "task_scores": {}, "run_names": {}, "curriculum_stage": stage}
    if state.get("curriculum_stage") != stage or state.get("max_iterations") != max_iterations:
        state.update(status="running", task_scores={}, current_task=None,
                     curriculum_stage=stage, max_iterations=max_iterations)
    if stage == "stage1":
        state["stage1_max_iterations"] = max_iterations
    state["phenotype_gate"] = str(root / "phenotype_gate.json")
    state["phenotype_fingerprint"] = gate["fingerprint"]
    _save(state_path, state)
    for env_id in sorted(ordered_tasks):
        task = TASKS[env_id]
        reference = _reference(task)
        probe_version=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
            ('mechanical_scene_contract.py','mechanical_joint_control.py','evaluate_mechanical_task.py'))).hexdigest()[:12]
        probe = root / "task_probes" / f"{task}_stage2_{probe_version}"
        probe_report = probe / "physics_batch_probe.json"
        if not probe_report.is_file() or not (
            json.loads(probe_report.read_text()).get("passed")
            and json.loads(probe_report.read_text()).get("motion_probe") == "bounded_sinusoidal_v1"
        ):
            probe.mkdir(parents=True, exist_ok=True)
            command = _base_command(task, root, stage, reference, probe)
            _run(command + ["--physics_batch_probe_envs", "4"], probe / "process.log")
            if not probe_report.is_file() or not json.loads(probe_report.read_text()).get("passed"):
                raise ValueError(f"Mechanical {task} scene probe did not pass")
    for env_id in sorted(requested_tasks):
        if env_id in state["task_scores"]:
            continue
        task = TASKS[env_id]
        run_name = _make_task_run_name(experiment_name, individual_id, env_id, stage)
        run_dir = Path(os.environ.get("EVOLUTION_ROOT", str(Path.home() / "Evolution_PC"))) / "evolution_tasks" / "logs" / "evolution_task" / run_name
        trial = output / "individuals" / individual_id / stage / task
        trial.mkdir(parents=True, exist_ok=True)
        state["run_names"][f"{stage}:{env_id}"] = run_name
        state["current_task"] = env_id
        _save(state_path, state)
        reference = _reference(task)
        source = reference
        if stage == "stage2":
            previous = state["run_names"].get(f"stage1:{env_id}")
            if previous:
                stage1 = run_dir.parent / previous / "nn" / "evolution_task.pth"
                if stage1.is_file():
                    source = stage1
            if source == reference:
                raise FileNotFoundError(f"Stage2 requires this individual's stage1 mechanical checkpoint: {env_id}")
        elif inherited_checkpoint_path and Path(inherited_checkpoint_path).is_file():
            parent_contract = Path(inherited_checkpoint_path).parent.parent / "params" / "policy_contract.json"
            if parent_contract.is_file():
                contract = json.loads(parent_contract.read_text())
                if contract.get("mechanical_adapter") and contract.get("task") == env_id:
                    source = Path(inherited_checkpoint_path)
        if parent_individual_id and source == reference:
            state.setdefault("policy_initialization", {})[f"{stage}:{env_id}"] = {
                "mode": "verified_BC_seed" if os.environ.get("EVOLUTION_MECHANICAL_BC_SEEDS") else "scratch_template", "parent_individual_id": parent_individual_id,
                "reason": "Task-specific parent unavailable; use verified task BC seed when configured"}
            _save(state_path, state)
        train_dir = trial / "train"
        trained = train_dir / "ppo" / "nn" / "mechanical_final.pth"
        training_report = train_dir / "mechanical_training.json"
        if not trained.is_file() or not training_report.is_file() or json.loads(training_report.read_text()).get("status") != "completed":
            train_dir.mkdir(exist_ok=True)
            command = _base_command(task, root, stage, source, train_dir)
            iterations = (max_iterations - state.get("stage1_max_iterations", 0)
                          if stage == "stage2" else max_iterations)
            if iterations <= 0:
                raise ValueError("Stage2 must extend stage1 iteration count")
            command += ["--train_iterations", str(iterations),
                        "--train_num_envs", str(train_envs),
                        "--train_from_scratch" if source == reference and not os.environ.get("EVOLUTION_MECHANICAL_BC_SEEDS") else "--train_auto_inherit"]
            _run(command, trial / "train.log")
        if not trained.is_file():
            raise FileNotFoundError(trained)
        training = json.loads(training_report.read_text())
        state.setdefault("policy_initialization", {})[f"{stage}:{env_id}"] = {
            "mode": "inherit" if training["weight_inheritance"] else "scratch",
            "source_checkpoint": training["source_checkpoint"],
            "parent_individual_id": parent_individual_id,
        }
        _save(state_path, state)
        (run_dir / "nn").mkdir(parents=True, exist_ok=True)
        (run_dir / "params").mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained, run_dir / "nn" / "evolution_task.pth")
        for name in ("agent.yaml", "policy_contract.json"):
            shutil.copy2(train_dir / "ppo" / "params" / name, run_dir / "params" / name)
        scores = []
        for seed in range(7, 7 + int(os.environ.get("EVOLUTION_MECHANICAL_EVAL_EPISODES", "3"))):
            eval_dir = trial / f"eval_seed{seed}"
            result_path = eval_dir / "evaluation.json"
            if not result_path.is_file():
                eval_dir.mkdir(exist_ok=True)
                command = _base_command(task, root, stage, trained, eval_dir)
                _run(command + ["--seed", str(seed)], eval_dir / "process.log")
            scores.append(_score(json.loads(result_path.read_text())))
        state["task_scores"][env_id] = sum(scores) / len(scores)
        state["current_task"] = None
        _save(state_path, state)
    state["status"] = "completed" if set(ordered_tasks).issubset(state["task_scores"]) else "running"
    _save(state_path, state)
    return _fitness_from_task_scores(state["task_scores"])
