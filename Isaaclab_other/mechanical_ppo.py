"""PPO warm-start through the same semantic thumb transform as evaluation."""
from pathlib import Path
import hashlib
import json
import os


def train_mechanical(raw_env, transfer, checkpoint, output_dir, iterations, smoke=False, inherit_weights=True, bc_manifest=None, bc_only=False, task=None):
    import gymnasium as gym
    import torch
    import yaml
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    class SemanticPolicyEnv(RlGamesVecEnvWrapper):
        def _process_obs(self, observations):
            # Evaluation clips physical observations before semantic transfer.
            # Keep this order identical, including scaled thumb velocities.
            clipped = super()._process_obs(observations)
            if isinstance(clipped, dict):
                raise ValueError('Asymmetric critic requires an explicit separate mapping')
            return transfer.observation_for_source_policy(clipped)

        def step(self, actions):
            return super().step(transfer.action_for_target_hand(actions))

    output_dir = Path(output_dir).resolve()
    source_params = Path(checkpoint).parent.parent/'params/agent.yaml'
    config = yaml.safe_load(source_params.read_text())
    params = config['params']; cfg = params['config']
    original = dict(cfg)
    cfg.update(num_actors=raw_env.unwrapped.num_envs, max_epochs=iterations,
               train_dir=str(output_dir), full_experiment_name='ppo',
               save_frequency=1 if smoke else cfg['save_frequency'])
    for key,variable in (('horizon_length','EVOLUTION_PPO_HORIZON_LENGTH'),
                         ('minibatch_size','EVOLUTION_PPO_MINIBATCH_SIZE'),
                         ('mini_epochs','EVOLUTION_PPO_MINI_EPOCHS')):
        if variable in os.environ:
            cfg[key]=int(os.environ[variable])
    batch = cfg['num_actors']*cfg['horizon_length']
    cfg['minibatch_size'] = min(cfg['minibatch_size'], batch)
    if batch % cfg['minibatch_size']:
        raise ValueError(f'PPO batch {batch} must divide minibatch {cfg["minibatch_size"]}')
    params['load_checkpoint'] = False
    params['load_path'] = ''
    wrapped = SemanticPolicyEnv(raw_env,cfg['device'],
        params['env']['clip_observations'],params['env']['clip_actions'])
    vecenv.register('IsaacRlgWrapper',lambda config_name,num_actors,**kwargs:RlGamesGpuEnv(config_name,num_actors,**kwargs))
    env_configurations.register('rlgpu',{'vecenv_type':'IsaacRlgWrapper','env_creator':lambda **kwargs:wrapped})
    runner = Runner(IsaacAlgoObserver()); runner.load(config); runner.reset()
    agent = runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
    from policy_inheritance import policy_contract as make_contract, map_model_weights
    raw = raw_env.unwrapped
    parent_contract = json.loads((source_params.parent/'policy_contract.json').read_text())
    policy_contract = make_contract(parent_contract['task'],raw.cfg,list(raw.hand.joint_names))
    policy_contract['actual_joint_names'] = list(raw.hand.joint_names)
    controller = getattr(raw,'mechanical_controller_name',policy_contract['controller'])
    if controller in ('grasp_joint_target_v1','forage_joint_target_v1','branch_joint_target_v1'):
        from mechanical_joint_control import layout
        actions=[]
        for finger,ids in enumerate(layout(raw)):
            actions.extend([f'joint:{raw.hand.joint_names[j]}.target' for j in ids])
            actions.extend([f'unused.finger{finger}.slot{i}' for i in range(len(ids),4)])
        if task == 'forage':
            actions += [f'wrist.delta.{axis}' for axis in 'xyz']
        policy_contract['observations'] = [name for name in policy_contract['observations'] if not name.startswith('previous_action:')] + ['previous_action:'+name for name in actions]
        policy_contract.update(controller=controller,actions=actions)
    source = torch.load(checkpoint,map_location='cpu',weights_only=False) if inherit_weights else None
    inheritance = None
    if source is not None:
        mapped, inheritance = map_model_weights(source['model'],agent.model.state_dict(),parent_contract,policy_contract)
        agent.model.load_state_dict(mapped,strict=True)
    report = {'source_checkpoint':str(checkpoint),
        'source_checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'source_agent_config_sha256':hashlib.sha256(source_params.read_bytes()).hexdigest(),
        'semantic_mapping':transfer.__dict__,'weight_inheritance':inherit_weights,
        'optimizer_reset':True,'epoch_reset':True,'named_inheritance':inheritance,
        'smoke_only':smoke,'config_changes':{k:{'before':original.get(k),'after':v} for k,v in cfg.items() if original.get(k)!=v},
        'status':'started'}
    report_path=output_dir/'mechanical_training.json'
    report_path.write_text(json.dumps(report,indent=2))
    (output_dir/'effective_agent.yaml').write_text(yaml.safe_dump(config))
    saved_params = output_dir/'ppo/params'
    saved_params.mkdir(parents=True,exist_ok=True)
    saved_config=yaml.safe_load(yaml.safe_dump(config))
    if bc_only:
        saved_config['params']['config']['minibatch_size']=original['minibatch_size']
    (saved_params/'agent.yaml').write_text(yaml.safe_dump(saved_config))
    assembly = json.loads((output_dir/'palm_assembly_contract.json').read_text())
    policy_contract['mechanical_adapter'] = {
        'observation_convention':'legacy_semantic_after_physical_clip',
        'mapping':transfer.__dict__,
        'rigid_palm_mode':assembly['mode'],
        'wrist_speed_m_s':assembly['mechanical_wrist_speed_m_s'],
        'target_urdf_sha256':hashlib.sha256(Path(raw_env.unwrapped.hand.cfg.spawn.asset_path).read_bytes()).hexdigest()}
    policy_contract['mechanical_scene']=json.loads((output_dir/'mechanical_scene_contract.json').read_text())
    (saved_params/'policy_contract.json').write_text(json.dumps(policy_contract,indent=2))
    # Checkpoint weights and input statistics are reused; optimizer and epoch
    # restart because the physical morphology and controller contract changed.
    if bc_manifest:
        from mechanical_bc import pretrain
        report['behavior_cloning']=pretrain(agent,transfer,bc_manifest,task,
            raw.hand.cfg.spawn.asset_path,policy_contract['mechanical_scene'],output_dir,
            params['env']['clip_observations'])
        report_path.write_text(json.dumps(report,indent=2))
    if bc_only and not bc_manifest:
        raise ValueError('BC-only initialization requires a verified manifest')
    if not bc_only:
        agent.train()
    current = agent.model.state_dict()
    report['all_model_tensors_finite'] = all(bool(torch.isfinite(v).all()) for v in current.values())
    report['changed_model_tensors'] = (sum(k not in source['model'] or not torch.equal(v.detach().cpu(),source['model'][k]) for k,v in current.items())
                                       if source is not None else None)
    if not report['all_model_tensors_finite'] or (source is not None and report['changed_model_tensors'] == 0):
        raise RuntimeError('PPO returned without finite updated model state')
    saved=output_dir/'ppo/nn/mechanical_final.pth'
    agent.save(str(saved.with_suffix('')))
    if not saved.is_file():
        raise RuntimeError(f'Training returned without checkpoint {saved}')
    report.update(status='completed',checkpoint=str(saved),bc_only=bc_only,epoch=int(agent.epoch_num),
                  checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    report_path.write_text(json.dumps(report,indent=2))
    print('[MECHANICAL_PPO_RESULT]',json.dumps(report),flush=True)
    wrapped.close()
