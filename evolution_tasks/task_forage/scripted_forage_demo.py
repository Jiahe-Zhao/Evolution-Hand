"""Record a deterministic, physically executed two-leaf Forage demonstration."""
from __future__ import annotations
import argparse, json, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record scripted Forage demonstration.")
parser.add_argument("--output", required=True)
parser.add_argument("--metrics", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args(); args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.evolution_tasks.task_forage  # noqa: F401
from isaaclab_tasks.evolution_tasks.task_forage.forage_env_cfg import ForageEnvCfg

def action(step, device):
    alpha = min(1.0, max(0.0, (step - 20) / 55.0))
    result = torch.full((1, 19), -0.6, dtype=torch.float32, device=device)
    result[:, :3] = -0.6 + alpha * 1.5
    result[:, 3:] = -0.6 + alpha * ((-0.3 if step >= 105 else 1.5))
    return result

def as_uint8(frame):
    frame = frame[0] if isinstance(frame, (tuple, list)) else frame
    frame = np.asarray(frame)
    return frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)

def main():
    cfg = ForageEnvCfg(); cfg.scene.num_envs = 1; cfg.scene.env_spacing = 2.0; cfg.reset_dof_pos_noise = 0.0; cfg.seed = 7
    # Both leaves begin covered (below 7.5 cm).  The scripted hand sweep must
    # move leaf one first, then leaf two, to obtain the original 300+700 reward.
    cfg.leaf_one_cfg.init_state.pos = (0.074, 0.0, 0.063)
    cfg.leaf_two_cfg.init_state.pos = (0.050, 0.002, 0.068)
    env = gym.make("Isaac-EvolutionHand-Forage-v0", cfg=cfg, render_mode="rgb_array"); env.reset(seed=7)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[0.28, -0.26, 0.42], target=[0.0, 0.0, 0.11], camera_prim_path="/OmniverseKit_Persp")
    except Exception as error: print(f"camera_view_warning={error}")
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    writer = imageio.get_writer(args_cli.output, fps=30, codec="libx264", quality=8)
    history=[]; seen_first=False; success=False
    try:
        for step in range(180):
            _, reward, terminated, truncated, _ = env.step(action(step, env.unwrapped.device))
            value=float(reward[0]); seen_first = seen_first or value >= 300.0
            # A 700-point event is impossible unless the 300-point first-leaf
            # latch has already been set by the environment.
            success = success or value >= 700.0
            env.unwrapped._compute_intermediate_values()
            food=env.unwrapped.food_pos[0,:2]
            d1=float(torch.norm(env.unwrapped.leaf_one_pos[0,:2]-food)); d2=float(torch.norm(env.unwrapped.leaf_two_pos[0,:2]-food))
            history.append({"step":step,"reward":value,"leaf_one_distance_m":d1,"leaf_two_distance_m":d2,"terminated":bool(terminated[0]),"truncated":bool(truncated[0])})
            frame=env.render()
            if frame is not None: writer.append_data(as_uint8(frame))
            if success: break
    finally: writer.close()
    summary={"task":"Forage","success":success,"seen_first_leaf_reward":seen_first,"required_total_reward":1000,"max_reward":max(x["reward"] for x in history),"steps_executed":len(history),"trajectory":"flex leaf1 -> reverse four-finger sweep leaf2","history":history}
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics)), exist_ok=True); json.dump(summary,open(args_cli.metrics,"w"),indent=2); print(json.dumps({k:v for k,v in summary.items() if k != "history"},indent=2))
    env.close(); simulation_app.close()
if __name__ == "__main__": main()
