"""Observe physical joint limits without changing PPO actions, rewards or resets."""
from types import MethodType
import torch


def attach_joint_quality_monitor(env, tolerance_rad=0.02):
    original = env._get_rewards
    state = {"max_rad": torch.zeros((), device=env.device),
             "violating_env_steps": torch.zeros((), device=env.device, dtype=torch.int64),
             "observed_env_steps": 0, "tolerance_rad": tolerance_rad}
    env.joint_quality_monitor = state

    def watched_rewards(self):
        rewards = original()
        joints = self.hand.data.joint_pos
        limits = self.hand.root_physx_view.get_dof_limits().to(joints.device)
        overshoot = torch.maximum(limits[..., 0] - joints, joints - limits[..., 1]).clamp_min(0)
        peak = overshoot.max(dim=-1).values
        state["max_rad"] = torch.maximum(state["max_rad"], peak.max())
        state["violating_env_steps"] += (peak > tolerance_rad).sum()
        state["observed_env_steps"] += joints.shape[0]
        log = self.extras.setdefault("log", {})
        log["physics/joint_overshoot_max_rad"] = peak.max()
        log["physics/joint_overshoot_env_fraction"] = (peak > tolerance_rad).float().mean()
        return rewards

    env._get_rewards = MethodType(watched_rewards, env)


def monitor_summary(env):
    state = env.joint_quality_monitor
    count = state["observed_env_steps"]
    violating = int(state["violating_env_steps"].item())
    return {"joint_limit_tolerance_rad": state["tolerance_rad"],
            "max_joint_overshoot_rad": float(state["max_rad"].item()),
            "violating_env_steps": violating, "observed_env_steps": count,
            "violating_env_step_fraction": violating / count if count else None,
            "role": "monitor_only; final checkpoint requires independent physical audit"}
