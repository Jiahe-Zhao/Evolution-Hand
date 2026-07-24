"""Persistent IsaacLab worker serving sequential evolution evaluation requests."""

import argparse
import copy
import gc
import glob
import importlib
import json
import math
import os
import random
import re
import time
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Persistent IsaacLab evolution worker.")
parser.add_argument("--request-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim must start before importing the rest of IsaacLab.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.timeline
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaacsim.core.utils.stage as stage_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


EVOLUTION_LOG_ROOT = os.environ.get(
    "EVOLUTION_LOG_ROOT", os.path.join(os.path.expanduser("~"), "Evolution_PC", "evolution_tasks", "logs")
)
TASK_MODULES = {
    "Isaac-EvolutionHand-BranchGrasp-v0": "isaaclab_tasks.evolution_tasks.task_branch_grasp.branch_grasp_env_cfg",
    "Isaac-EvolutionHand-Carry-v0": "isaaclab_tasks.evolution_tasks.task_carry.carry_env_cfg",
    "Isaac-EvolutionHand-Forage-v0": "isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg",
    "Isaac-EvolutionHand-Grasp-v0": "isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg",
    "Isaac-EvolutionHand-Manipulation-v0": "isaaclab_tasks.evolution_tasks.task_manipulation.evolution_manipulation_env_cfg",
    "Isaac-EvolutionHand-Strike-v0": "isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg",
    "Isaac-EvolutionHand-StoneGrind-v0": "isaaclab_tasks.evolution_tasks.task_stone.evolution_stone_grind_env_cfg",
}
HAND_MODULES = (
    "isaaclab_tasks.evolution_tasks.current_right_hand.current_right_hand_cfg",
    "isaaclab_tasks.evolution_tasks.current_left_hand.current_left_hand_cfg",
)
KEEP_LATEST_CHECKPOINTS = max(1, int(os.environ.get("EVOLUTION_KEEP_LATEST_CHECKPOINTS", "1")))
KEEP_BEST_CHECKPOINTS = max(1, int(os.environ.get("EVOLUTION_KEEP_BEST_CHECKPOINTS", "1")))
CHECKPOINT_REWARD_PATTERN = re.compile(r"rew_([-+]?\d*\.?\d+|\d+)")
CHECKPOINT_EPOCH_PATTERN = re.compile(r"_ep_(\d+)")


def _atomic_write_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _reload_generated_configs(task_name):
    task_module = TASK_MODULES.get(task_name)
    if task_module is None:
        raise ValueError(f"Unsupported task: {task_name}")
    importlib.invalidate_caches()
    for module_name in (*HAND_MODULES, task_module):
        module = importlib.import_module(module_name)
        importlib.reload(module)


def _prepare_clean_stage():
    """Reset the stage after the previous task while retaining the Isaac application."""
    SimulationContext.clear_instance()
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()
    for _ in range(3):
        simulation_app.update()
    stage_utils.create_new_stage()
    for _ in range(3):
        simulation_app.update()


def _checkpoint_reward(path):
    match = CHECKPOINT_REWARD_PATTERN.search(os.path.basename(path))
    return float(match.group(1)) if match else float("-inf")


def _checkpoint_epoch(path):
    match = CHECKPOINT_EPOCH_PATTERN.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def _prune_run_checkpoints(run_dir):
    nn_dir = os.path.join(run_dir, "nn")
    checkpoints = glob.glob(os.path.join(nn_dir, "*.pth")) if os.path.isdir(nn_dir) else []
    if len(checkpoints) <= 1:
        return
    latest = sorted(checkpoints, key=os.path.getmtime, reverse=True)[:KEEP_LATEST_CHECKPOINTS]
    best = sorted(checkpoints, key=_checkpoint_reward, reverse=True)[:KEEP_BEST_CHECKPOINTS]
    for path in checkpoints:
        if path not in set(latest) | set(best):
            os.remove(path)


def _run_training(request):
    task_name = request["task"]
    _prepare_clean_stage()
    _reload_generated_configs(task_name)

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")
    agent_cfg = copy.deepcopy(agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg)
    env_cfg.scene.num_envs = int(request["num_envs"])
    env_cfg.sim.device = request["device"]
    # Keep physics and the RL policy on the same Slurm-assigned GPU slot.
    agent_cfg["params"]["config"]["device"] = request["device"]
    env_cfg.seed = request.get("seed", agent_cfg["params"]["seed"])
    if env_cfg.seed == -1:
        env_cfg.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = env_cfg.seed
    if request.get("max_iterations") is not None:
        agent_cfg["params"]["config"]["max_epochs"] = int(request["max_iterations"])
    if request.get("checkpoint_interval", 0) > 0:
        agent_cfg["params"]["config"]["save_frequency"] = int(request["checkpoint_interval"])

    run_name = request["run_name"]
    log_root = os.path.abspath(os.path.join(EVOLUTION_LOG_ROOT, agent_cfg["params"]["config"]["name"]))
    params_dir = os.path.join(log_root, run_name, "params")
    os.makedirs(params_dir, exist_ok=True)
    agent_cfg["params"]["config"]["train_dir"] = log_root
    agent_cfg["params"]["config"]["full_experiment_name"] = run_name
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)
    dump_pickle(os.path.join(params_dir, "agent.pkl"), agent_cfg)

    checkpoint_path = request.get("checkpoint_path")
    resume_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        resume_path = retrieve_file_path(checkpoint_path)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = None
    runner = None
    try:
        env = gym.make(task_name, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()
        run_args = {"train": True, "play": False}
        if resume_path:
            run_args["checkpoint"] = resume_path
        runner.run(run_args)
        run_dir = os.path.join(log_root, run_name)
        os.makedirs(os.path.join(run_dir, "finished"), exist_ok=True)
        _prune_run_checkpoints(run_dir)
        return {"run_dir": run_dir}
    finally:
        if env is not None:
            env.close()
        runner = None
        env = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    request_dir = os.path.abspath(args_cli.request_dir)
    os.makedirs(request_dir, exist_ok=True)
    _atomic_write_json(os.path.join(request_dir, "ready.json"), {"pid": os.getpid(), "ready_at": time.time()})
    while True:
        request_files = sorted(name for name in os.listdir(request_dir) if name.endswith(".request.json"))
        if not request_files:
            time.sleep(0.2)
            continue
        request_path = os.path.join(request_dir, request_files[0])
        working_path = request_path.replace(".request.json", ".working.json")
        try:
            os.replace(request_path, working_path)
        except FileNotFoundError:
            continue
        request = {}
        try:
            with open(working_path, "r", encoding="utf-8") as file:
                request = json.load(file)
            request_id = request["id"]
            if request.get("command") == "shutdown":
                _atomic_write_json(os.path.join(request_dir, f"{request_id}.response.json"), {"ok": True})
                break
            print(f"[WORKER] Starting request {request_id}: {request['task']}", flush=True)
            response = {"ok": True, **_run_training(request)}
            print(f"[WORKER] Completed request {request_id}: {request['task']}", flush=True)
        except Exception as error:  # noqa: BLE001
            response = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
            print(f"[WORKER] Request {request.get('id', 'unknown')} failed: {error}", flush=True)
        _atomic_write_json(os.path.join(request_dir, f"{request.get('id', 'unknown')}.response.json"), response)
        os.remove(working_path)
    simulation_app.close()


if __name__ == "__main__":
    main()
