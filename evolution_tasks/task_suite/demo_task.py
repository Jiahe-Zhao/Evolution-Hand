"""Record a standardized scripted demonstration for one Evolution task."""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

from task_registry import TASKS

CAMERA_VIEWS = {
    "grasp": ([0.42, -0.42, 0.58], [0.0, 0.0, 0.26]),
    "branch": ([-0.34, -0.46, 0.58], [0.0, 0.0, 0.29]),
    "forage": ([0.36, -0.36, 0.42], [0.0, 0.0, 0.11]),
    "strike": ([-0.55, -0.50, 0.60], [-0.05, 0.01, 0.23]),
}
DEMO_ENV_INDEX = {"grasp": 0, "branch": 0, "forage": 1, "strike": 0}

parser = argparse.ArgumentParser(description="Record a standardized Evolution task demonstration.")
parser.add_argument("--task", choices=TASKS, required=True)
parser.add_argument("--output", required=True, help="MP4 output path.")
parser.add_argument("--metrics", required=True, help="JSON metrics output path.")
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--success_reward", type=float, default=1000.0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import importlib
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401


def _frame_u8(frame):
    if isinstance(frame, (tuple, list)):
        frame = frame[0]
    return np.asarray(frame).astype(np.uint8)


def _action(task: str, step: int, action_dim: int, device: torch.device) -> torch.Tensor:
    action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
    if task == "grasp":
        # Maintain the pre-grasp briefly, then close around the finger-pad load.
        if step < 8:
            action[:] = -0.30
        elif step < 28:
            action[:] = -0.30 + 1.20 * (step - 8) / 20.0
        else:
            action[:] = 0.90
    elif task == "branch":
        # Validated branch pinch: settle open, then close all digits together
        # and maintain the required contact-force hold.
        if step < 30:
            action[:] = -0.70
        elif step < 90:
            action[:] = -0.70 + 1.45 * (step - 30) / 60.0
        else:
            action[:] = 0.90
    elif task == "forage":
        # Close the fingers while approaching, then sweep laterally using the
        # three wrist action channels added to the forage environment.
        alpha = min(1.0, max(0.0, (step - 20) / 55.0))
        action[:] = -0.60
        action[:, :3] = -0.60 - 0.30 * alpha
        action[:, 3:19] = -0.60 + (1.50 if step < 105 else -0.30) * alpha
        approach = min(1.0, max(0.0, (step - 20) / 50.0))
        sweep = min(1.0, max(0.0, (step - 85) / 45.0))
        action[:, 19] = -0.35 + 1.20 * sweep  # x: push leaves off the food.
        action[:, 20] = -0.95 + 1.80 * approach  # y: approach from the front.
        action[:, 21] = -0.75  # z: lower palm toward the leaf layer.
    # Strike starts from a prescribed pre-grasp.  Its final action channel is
    # wrist z, so this produces the same hold-then-strike trajectory used in
    # the validated scene demonstration.
    if task == "strike" and step >= 45:
        action[:, -1] = -min(1.0, (step - 44) / 35.0)
    return action


def _state(task: str, step: int) -> str:
    """Return the named phase of each deterministic demonstration."""
    if task == "grasp":
        return "support" if step < 8 else "close" if step < 28 else "hold"
    if task == "branch":
        return "open" if step < 30 else "close" if step < 90 else "hold"
    if task == "forage":
        return "prepare" if step < 20 else "approach" if step < 85 else "sweep"
    return "pregrasp" if step < 45 else "strike"


def main():
    env_id, module_name, cfg_module_name, cfg_name = TASKS[args.task]
    if args.task == "branch":
        os.environ["EVOLUTION_CURRICULUM_STAGE"] = "stage1"
    importlib.import_module(module_name)
    cfg_type = getattr(importlib.import_module(cfg_module_name), cfg_name)
    cfg = cfg_type()
    cfg.scene.num_envs = 6 if args.task == "forage" else 1
    cfg.seed = 7
    if hasattr(cfg, "reset_dof_pos_noise"):
        cfg.reset_dof_pos_noise = 0.0
    if args.task == "grasp":
        # Pre-grasp: curl all fingers before physics starts, with the ball in
        # the finger-pad envelope rather than resting on the distal-link backs.
        cfg.robot_cfg.init_state.joint_pos = {".*": 0.35}
        cfg.grasp_object_cfg.init_state.pos = (0.010, 0.005, 0.365)
    elif args.task == "branch":
        cfg.branch_cfg.init_state.pos = (0.0, 0.012, 0.30)
    elif args.task == "forage":
        # Rest on top of the food support (top z=0.100), never inside it.
        # The leaves start covered but close enough for the sweep to clear.
        cfg.leaf_one_cfg.init_state.pos = (0.070, 0.0, 0.1022)
        cfg.leaf_two_cfg.init_state.pos = (0.060, 0.002, 0.1064)
        # Bring the palm near the leaf layer while keeping the leaves on their
        # support, so clearance comes from the scripted finger sweep.
        cfg.robot_cfg.init_state.pos = (0.020, -0.080, 0.115)
    eye, target = CAMERA_VIEWS[args.task]
    # env.render() uses the IsaacLab viewer configuration, not only the UI viewport.
    cfg.viewer.eye = tuple(eye)
    cfg.viewer.lookat = tuple(target)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = DEMO_ENV_INDEX[args.task]

    env = gym.make(env_id, cfg=cfg, render_mode="rgb_array")
    env.reset(seed=7)
    demo_env_index = DEMO_ENV_INDEX[args.task]
    proximal_support_body_ids = []
    locked_base_action_indices = []
    locked_base_actions = None
    if args.task == "grasp":
        support_links = ("link_2_0", "link_3_0", "link_4_0")
        proximal_support_body_ids = [env.unwrapped.hand.body_names.index(name) for name in support_links]
        support_center_w = env.unwrapped.hand.data.body_pos_w[demo_env_index, proximal_support_body_ids].mean(dim=0)
        object_state = env.unwrapped.grasp_object.data.default_root_state[demo_env_index : demo_env_index + 1].clone()
        object_state[:, 0:3] = support_center_w + torch.tensor([0.0, 0.0, 0.030], device=env.unwrapped.device)
        object_state[:, 7:] = 0.0
        demo_env_ids = torch.tensor([demo_env_index], dtype=torch.long, device=env.unwrapped.device)
        env.unwrapped.grasp_object.write_root_state_to_sim(object_state, demo_env_ids)

        base_joint_names = ("link_0_0_to_link_2_0", "link_0_0_to_link_3_0", "link_0_0_to_link_4_0", "link_0_0_to_link_5_0")
        joint_ids = env.unwrapped.actuated_dof_indices
        locked_base_action_indices = [joint_ids.index(env.unwrapped.hand.joint_names.index(name)) for name in base_joint_names]
        lower = env.unwrapped.hand_dof_lower_limits[demo_env_index, joint_ids]
        upper = env.unwrapped.hand_dof_upper_limits[demo_env_index, joint_ids]
        default = env.unwrapped.hand.data.default_joint_pos[demo_env_index, joint_ids]
        locked_base_actions = 2.0 * (default - lower) / (upper - lower) - 1.0
        env.unwrapped._compute_intermediate_values()
    initial_joint_pos = env.unwrapped.hand.data.joint_pos[demo_env_index].clone()
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
    successes, max_reward = 0, 0.0
    scripted_successes, proximal_hold_steps = 0, 0
    history = []
    try:
        for step in range(args.steps):
            action = _action(args.task, step, env.unwrapped.cfg.action_space, env.unwrapped.device)
            if args.task == "grasp":
                # Palm is fixed by the articulation root; lock long-finger base
                # joints so their first phalanges remain a stable support shelf.
                action[0, locked_base_action_indices] = locked_base_actions[locked_base_action_indices]
            if args.task == "forage":
                action = action.repeat(env.unwrapped.num_envs, 1)
            _, reward, terminated, truncated, _ = env.step(action)
            value = float(reward[demo_env_index])
            max_reward = max(max_reward, value)
            successes += int(value >= args.success_reward)
            record = {"step": step, "state": _state(args.task, step), "reward": value}
            current_joint_pos = env.unwrapped.hand.data.joint_pos[demo_env_index]
            record["mean_abs_joint_displacement_rad"] = float(
                torch.mean(torch.abs(current_joint_pos - initial_joint_pos))
            )
            if args.task == "grasp":
                record["object_in_palm"] = bool(env.unwrapped.object_in_palm[demo_env_index])
                record["object_in_distal_finger_region"] = bool(
                    env.unwrapped.object_in_distal_finger_region[demo_env_index]
                )
                record["object_in_grasp_region"] = bool(env.unwrapped.object_in_grasp_region[demo_env_index])
                record["vertical_force_n"] = float(env.unwrapped.grasp_object_force[demo_env_index, 2])
                record["success_hold_steps"] = int(env.unwrapped.success_streaks[demo_env_index])
                object_pos = env.unwrapped.grasp_object_pos[demo_env_index]
                proximal_pos = env.unwrapped.hand.data.body_pos_w[demo_env_index, proximal_support_body_ids] - env.unwrapped.scene.env_origins[demo_env_index]
                proximal_distances = torch.norm(proximal_pos - object_pos.unsqueeze(0), dim=-1)
                in_proximal_shelf = int(torch.sum(proximal_distances <= 0.060)) >= 2
                force_ok = abs(record["vertical_force_n"]) >= 7.0
                proximal_hold_steps = proximal_hold_steps + 1 if in_proximal_shelf and force_ok else 0
                scripted_success = proximal_hold_steps >= 10
                scripted_successes += int(scripted_success)
                record["object_in_proximal_shelf"] = in_proximal_shelf
                record["proximal_shelf_hold_steps"] = proximal_hold_steps
                record["scripted_success"] = scripted_success
            elif args.task == "branch":
                scores = env.unwrapped.finger_action_scores[demo_env_index]
                record["long_finger_flexion_scores"] = [float(score) for score in scores[1:]]
                record["long_finger_flexion_spread"] = float(scores[1:].max() - scores[1:].min())
                record["long_finger_velocity_spread"] = float(
                    env.unwrapped.long_finger_velocity_spread[demo_env_index]
                )
                record["long_finger_joint_velocity_scores"] = [
                    float(score) for score in env.unwrapped.long_finger_joint_velocity_scores[demo_env_index]
                ]
                record["long_finger_joint_scores"] = [
                    float(score) for score in env.unwrapped.long_finger_joint_scores[demo_env_index]
                ]
                record["long_finger_joint_velocity_spread"] = float(
                    env.unwrapped.long_finger_joint_velocity_spread[demo_env_index]
                )
                joint_ids = env.unwrapped.actuated_dof_indices
                raw_joint_pos = env.unwrapped.hand.data.joint_pos[demo_env_index, joint_ids]
                target_joint_pos = env.unwrapped.cur_targets[demo_env_index, joint_ids]
                record["long_finger_raw_joint_scores"] = [
                    float(raw_joint_pos[start:stop].mean())
                    for start, stop in ((3, 7), (7, 11), (11, 15), (15, 19))
                ]
                record["long_finger_target_joint_scores"] = [
                    float(target_joint_pos[start:stop].mean())
                    for start, stop in ((3, 7), (7, 11), (11, 15), (15, 19))
                ]
            elif args.task == "forage":
                if value >= args.success_reward:
                    distances = env.unwrapped.success_leaf_distances[demo_env_index]
                    record["success_leaf_one_distance_m"] = float(distances[0])
                    record["success_leaf_two_distance_m"] = float(distances[1])
            history.append(record)
            writer.append_data(_frame_u8(env.render()))
            if terminated[demo_env_index] or truncated[demo_env_index]:
                break
    finally:
        writer.close()
        env.close()

    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(
            {
                "task": args.task,
                "success_rule": f"reward >= {args.success_reward}",
                "successes": successes,
                "scripted_successes": scripted_successes,
                "scripted_success_rule": "two proximal support links within 0.06 m, vertical force >= 7 N, held for 10 steps",
                "max_reward": max_reward,
                "steps_executed": len(history),
                "history": history,
                "scripted_wrist_policy": args.task == "forage",
            },
            file,
            indent=2,
        )
    app.close()


if __name__ == "__main__":
    main()
