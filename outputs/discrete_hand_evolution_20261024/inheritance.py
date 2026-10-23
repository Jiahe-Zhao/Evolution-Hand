"""Generation-aware BC permissions and actor-only inheritance for rl_games."""
import hashlib
import json
import math
from pathlib import Path


def validate_request(generation, *, checkpoint='', mode='ppo', refit=False,
                     residual_bc=False, bc_anchor=0., collect=False, **kwargs):
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    if residual_bc or bc_anchor:
        raise ValueError("Frozen BC residuals and persistent BC anchors are disabled")
    if generation > 0 and (mode == 'bc' or refit or collect or not checkpoint):
        raise ValueError("Descendants require a compatible parent checkpoint and cannot run BC")
    return generation == 0 and (not checkpoint or refit) and mode != 'eval'


def policy_contract(names, limits):
    return dict(version=1, task='free_tool_wrap', observation='wrap67_static_scaling_v1',
                action='joint_absolute_normalized_and_world_wrist_0.25m',
                joint_names=list(names), joint_limits=limits,
                separate_actor_critic=True, normalize_input=False)


def check_seed_quality(generation, successes, episodes):
    """Only the founding seed must pass BC quality; descendants get RL budget."""
    if generation == 0 and successes < math.ceil(.9*episodes):
        raise ValueError('Generation-zero BC seed has not passed closed-loop quality validation')


def load_actor(agent, checkpoint, parent_contract, child_contract):
    """Copy the complete actor, keep fresh critic, optimizer and value statistics.

    A changed morphology is allowed. A changed task, encoder, joint order,
    limits, output meaning, or network shape is not silently accepted.
    """
    import torch
    if parent_contract != child_contract:
        raise ValueError("Parent policy interface is incompatible; no BC fallback is permitted")
    if agent.optimizer.state:
        raise ValueError("Actor inheritance must run before optimizer updates")
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location=agent.ppo_device, weights_only=False)['model']
    state = agent.model.state_dict()
    keys = [k for k in state if k.startswith(('a2c_network.actor_mlp.', 'a2c_network.mu.'))]
    if not keys or not any(k.startswith('a2c_network.mu.') for k in keys):
        raise ValueError("Expected a separate rl_games actor network")
    for key in keys:
        if key not in saved or saved[key].shape != state[key].shape:
            raise ValueError("Incompatible actor tensor: " + key)
        if not torch.isfinite(saved[key]).all():
            raise ValueError("Nonfinite parent actor: " + key)
        state[key] = saved[key].to(state[key])
    agent.model.load_state_dict(state, strict=True)
    if agent.optimizer.state:
        raise AssertionError("Optimizer state was inherited")
    error = max(float((agent.model.state_dict()[k]-saved[k].to(state[k])).abs().max()) for k in keys)
    return dict(mode='actor_only', parent_checkpoint=str(checkpoint.resolve()),
                parent_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                copied_keys=keys, actor_max_copy_error=error,
                critic='fresh', optimizer='fresh', value_normalizer='fresh',
                exploration='configured_for_child', bc_updates=0)


def read_contract(checkpoint):
    path = Path(checkpoint).parent / 'policy_contract.json'
    if not path.is_file():
        raise ValueError(f"Missing verified policy contract: {path}")
    return json.loads(path.read_text())
