"""Task-local PPO skill transfer through the existing Cartesian/IK interface."""
from pathlib import Path
import os
import json
import re

def distal_body_name(finger_id, available_joints):
    bodies = []
    for name in available_joints:
        match = re.search(r'_to_(link_' + str(finger_id) + r'_(\d+))$', name)
        if match:
            bodies.append((int(match[2]), match[1]))
    return max(bodies)[1] if bodies else None

def adapt_task_config(env_cfg, available_joints):
    """Resolve arbitrary finger chains without rewriting object-contact filters."""
    available = set(available_joints)
    initial = env_cfg.robot_cfg.init_state
    joint_pos = {k:v for k,v in (initial.joint_pos or {}).items() if k == '.*' or k in available}
    env_cfg.robot_cfg = env_cfg.robot_cfg.replace(init_state=initial.replace(joint_pos=joint_pos))
    tips = [name for f in range(1,6) if (name := distal_body_name(f,available))]
    if hasattr(env_cfg, 'fingertip_body_names'):
        env_cfg.fingertip_body_names = tips
    for attr in dir(env_cfg):
        if not attr.endswith('contact_sensor_cfg'):
            continue
        sensor = getattr(env_cfg, attr)
        paths = getattr(sensor, 'filter_prim_paths_expr', None) or []
        if paths and all(re.search(r'/link_[1-5]_\d+$', p) for p in paths):
            roots = {p.rsplit('/',1)[0] for p in paths}
            if len(roots) != 1:
                raise ValueError('Ambiguous hand contact-filter roots')
            root = roots.pop()
            sensor.filter_prim_paths_expr = [root+'/'+tip for tip in tips]
    return tips

def select_parent_checkpoint(parent, task, log_root, state=None):
    """Only select the actual parent's same-task policy, never another task."""
    metadata = (parent or {}).get('metadata', {})
    runs = dict((state or {}).get('run_names', {}))
    runs.update(metadata.get('stage1_run_names', {}))
    runs.update(metadata.get('stage2_run_names', {}))
    for stage in ('stage2', 'stage1'):
        run = runs.get(stage+':'+task)
        if not run:
            continue
        nn = Path(log_root)/run/'nn'
        best = nn/'evolution_task.pth'
        candidates = list(nn.glob('*.pth'))
        if best.is_file():
            return str(best)
        if candidates:
            return str(max(candidates, key=lambda p:p.stat().st_mtime))
    return None

TASKS = {'Isaac-EvolutionHand-Grasp-v0': 'grasp',
         'Isaac-EvolutionHand-BranchGrasp-v0': 'branch',
         'Isaac-EvolutionHand-Forage-v0': 'forage',
         'Isaac-EvolutionHand-Strike-v0': 'strike'}

def _field(cfg, name, default=None):
    return cfg.get(name, default) if isinstance(cfg, dict) else getattr(cfg, name, default)

def policy_contract(task, cfg, actual_joint_names=None):
    kind = TASKS[task]
    if str(_field(cfg, 'asymmetric_obs', False)).lower() == 'true':
        raise ValueError('Asymmetric critic transfer requires a separate observation contract')
    if _field(cfg, 'obs_type', 'full') != 'full':
        raise ValueError('Skill transfer requires the current full observation layout')
    joints = list(_field(cfg, 'actuated_joint_names'))
    branch_joint_mode = kind == 'branch' and os.environ.get('EVOLUTION_BRANCH_BC_MODE') == '1'
    strike_joint_mode = kind == 'strike' and os.environ.get('EVOLUTION_STRIKE_BC_MODE') == '1'
    if branch_joint_mode:
        actions = [f'finger{f}.joint{j}.target' for f in range(1, 6) for j in range(4)]
    elif strike_joint_mode:
        physical_joints = list(actual_joint_names) if actual_joint_names is not None else list(joints)
        if len(physical_joints) > 20:
            raise ValueError('Strike direct BC controller supports at most 20 physical joints')
        physical_joints += [f'unused_strike_joint_{index}' for index in range(len(physical_joints), 20)]
        actions = [f'joint:{name}.target' for name in physical_joints]
    else:
        actions = [f'finger{f}.delta.{axis}' for f in range(1, 6) for axis in 'xyz']
        actions += [f'finger{f}.closure' for f in range(1, 6)]
    if kind in ('forage', 'strike'):
        actions += [f'wrist.delta.{axis}' for axis in 'xyz']
    obs = [f'q:{j}' for j in joints] + [f'dq:{j}' for j in joints]
    tips = [f'tip{f}.position.{a}' for f in range(1, 6) for a in 'xyz']
    descriptor = [f'morph.tip{f}.{a}' for f in range(1, 6) for a in 'xyz']
    descriptor += [f'morph.finger{f}.active' for f in range(1, 6)]
    if kind in ('grasp', 'strike'):
        obs += [f'object.{field}.{i}' for field, n in [('pos',3),('quat',4),('linvel',3),('angvel',3),('force',3)] for i in range(n)]
        obs += tips
        obs += [f'tip{f}.quat.{i}' for f in range(1,6) for i in range(4)]
        obs += [f'tip{f}.velocity.{i}' for f in range(1,6) for i in range(6)]
    else:
        obs += tips
        objects = ['branch'] if kind == 'branch' else ['food','leaf_one','leaf_two']
        obs += [f'{obj}.pose.{i}' for obj in objects for i in range(7)]
    obs += descriptor + ['previous_action:'+a for a in actions]
    if len(actions) != int(_field(cfg, 'action_space')) or len(obs) != int(_field(cfg, 'observation_space')):
        raise ValueError(f'Unrecognized {task} action/observation layout')
    controller = 'branch_joint_target_v1' if branch_joint_mode else 'strike_joint_target_v1' if strike_joint_mode else 'cartesian_5finger_ik_v1'
    contract = {'version':1, 'task':task, 'controller':controller, 'actions':actions, 'observations':obs}
    if strike_joint_mode:
        contract['environment'] = {
            'EVOLUTION_STRIKE_RESET_THUMB_SPREAD': os.environ.get('EVOLUTION_STRIKE_RESET_THUMB_SPREAD', '-0.80'),
            'EVOLUTION_STRIKE_PREGRASP_ACTION': os.environ.get('EVOLUTION_STRIKE_PREGRASP_ACTION', str(_field(cfg, 'pregrasp_action', 0.45))),
            'EVOLUTION_STRIKE_RESET_OFFSET_WORLD': os.environ.get('EVOLUTION_STRIKE_RESET_OFFSET_WORLD', '0,0,0'),
        }
    return contract

def load_contract(checkpoint, task):
    params = Path(checkpoint).parent.parent/'params'
    path = params/'policy_contract.json'
    if path.exists():
        contract = json.loads(path.read_text())
    else:
        import yaml
        with (params/'env.yaml').open() as stream:
            cfg = yaml.load(stream, Loader=yaml.BaseLoader)
        contract = policy_contract(task, cfg)
    if contract['task'] != task:
        raise ValueError('Cannot inherit a policy from another task')
    return contract

def semantic_pairs(source, target):
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError('Duplicate policy channel names')
    indices = {name:i for i,name in enumerate(source)}
    return [(j, indices[name]) for j,name in enumerate(target) if name in indices]

def map_model_weights(source, target, parent, child):
    """Remap input columns, action rows, and normalization by semantic names."""
    if parent['task'] != child['task'] or parent['controller'] != child['controller']:
        raise ValueError('Task/controller contract mismatch')
    op = semantic_pairs(parent['observations'], child['observations'])
    ap = semantic_pairs(parent['actions'], child['actions'])
    if not op or not ap:
        raise ValueError('No transferable policy channels')
    mapped = {}
    for name, initial in target.items():
        if name not in source:
            raise ValueError('Parent model missing parameter '+name)
        value = source[name].to(device=initial.device, dtype=initial.dtype)
        dest = initial.clone()
        if name.endswith(('actor_mlp.0.weight','critic_mlp.0.weight')):
            if value.shape[0] != dest.shape[0] or value.shape[1] != len(parent['observations']) or dest.shape[1] != len(child['observations']):
                raise ValueError('Incompatible input layer '+name)
            dest.zero_()
            for j,i in op: dest[:,j] = value[:,i]
        elif name.endswith(('mu.weight','mu.bias','sigma')) or name.endswith(('sigma.weight','sigma.bias')):
            if value.shape[0] != len(parent['actions']) or dest.shape[0] != len(child['actions']) or value.shape[1:] != dest.shape[1:]:
                raise ValueError('Incompatible action layer '+name)
            for j,i in ap: dest[j] = value[i]
        elif 'running_mean_std' in name and name.endswith(('running_mean','running_var')):
            if value.numel() != len(parent['observations']) or dest.numel() != len(child['observations']):
                raise ValueError('Incompatible observation normalizer '+name)
            for j,i in op: dest[j] = value[i]
        elif value.shape == dest.shape:
            dest.copy_(value)
        else:
            raise ValueError('Incompatible hidden model parameter '+name)
        mapped[name] = dest
    return mapped, {'mapped_observations':len(op),'total_observations':len(child['observations']),
                    'mapped_actions':len(ap),'total_actions':len(child['actions'])}

def save_contract(run_dir, task, env):
    contract = policy_contract(task, env.cfg, env.hand.joint_names)
    if not hasattr(env, 'cartesian_ik'):
        raise ValueError('Expected morphology-aware Cartesian IK controller')
    contract['actual_joint_names'] = list(env.hand.joint_names)
    contract['fingertip_bodies'] = list(env.cartesian_ik.finger_names)
    path = Path(run_dir)/'params/policy_contract.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract,indent=2))
    return contract

def train_inherited(runner, checkpoint, contract, run_dir, *, start_training=True):
    import torch
    parent = load_contract(checkpoint, contract['task'])
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state, report = map_model_weights(source['model'], agent.model.state_dict(), parent, contract)
    agent.model.load_state_dict(state, strict=True)
    report.update(source_checkpoint=str(checkpoint),task=contract['task'],
                  mode='mapped_weights_only',optimizer_reset=True,epoch_reset=True,
                  actual_joint_names=contract['actual_joint_names'],
                  fingertip_bodies=contract['fingertip_bodies'])
    (Path(run_dir)/'inheritance.json').write_text(json.dumps(report,indent=2))
    print('[INHERIT] '+json.dumps(report),flush=True)
    if start_training:
        agent.train()
    return agent

def train_behavior_cloning(agent, dataset_path, run_dir, epochs=10, batch_size=256, learning_rate=1e-4):
    """Warm-start actor mean from a verified policy-action demonstration."""
    import numpy as np
    import torch
    data = np.load(dataset_path)
    observations = torch.as_tensor(data['observations_before_step'], dtype=torch.float32, device=agent.ppo_device)
    actions = torch.as_tensor(data['submitted_actions'], dtype=torch.float32, device=agent.ppo_device)
    if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions):
        raise ValueError(f'Invalid BC dataset shapes: {observations.shape}, {actions.shape}')
    if len(observations) == 0 or not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError('BC dataset is empty or contains non-finite values')
    if 'actions_control_fingers' not in data or not bool(np.asarray(data['actions_control_fingers']).all()):
        raise ValueError('BC dataset contains steps driven by a scripted joint override')
    if 'scene_unmodified' not in data or not bool(np.asarray(data['scene_unmodified']).all()):
        raise ValueError('BC dataset contains scripted changes to the task scene')
    if actions.shape[1] < 20 or not bool((actions[:, :20].abs().amax(dim=0) > 1e-4).any()):
        raise ValueError('BC dataset contains no fingertip actions')
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)
    agent.model.train()
    losses = []
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(len(observations), device=observations.device)
        for start in range(0, len(order), max(1, int(batch_size))):
            ids = order[start:start + batch_size]
            result = agent.model({'is_train': True, 'prev_actions': actions[ids], 'obs': observations[ids]})
            loss = torch.nn.functional.mse_loss(result['mus'], actions[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    report = {'mode': 'behavior_cloning', 'dataset': str(dataset_path), 'epochs': int(epochs), 'samples': len(observations), 'final_loss': losses[-1] if losses else None}
    (Path(run_dir) / 'behavior_cloning.json').write_text(json.dumps(report, indent=2))
    print('[BC] ' + json.dumps(report), flush=True)
    return report
