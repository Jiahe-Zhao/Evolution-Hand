"""Strict successful-demonstration BC for mechanical-hand policies."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_demonstrations(manifest_path, task, urdf, scene):
    import numpy as np
    manifest=json.loads(Path(manifest_path).read_text())
    if set(manifest['tasks']) != {'grasp','branch','forage','strike'}:
        raise ValueError('All four tasks must have verified demonstrations before training')
    # Verify every task's reports, so a partial manifest cannot start the run.
    for key, entries in manifest['tasks'].items():
        if not entries:
            raise ValueError(f'No demonstrations for {key}')
        for entry in entries:
            for field in ('metrics','trace','scene'):
                if sha256(entry[field]) != entry[field+'_sha256']:
                    raise ValueError(f'Demonstration file changed: {entry[field]}')
            report=json.loads(Path(entry['metrics']).read_text())
            if not report['success'] or report.get('joint_limit_violation_steps',0):
                raise ValueError(f'Unsuccessful or joint-violating demonstration: {key}')
            if key == 'grasp' and (not report.get('environment_m3_success') or report.get('source_mesh_penetration_steps',0)):
                raise ValueError('Grasp requires native M3 success without source-mesh penetration')
            if not report.get('training_scene'):
                raise ValueError('Demonstration did not use the native task reset')
    observations,actions=[],[]
    for entry in manifest['tasks'][task]:
        contract=json.loads(Path(entry['scene']).read_text())
        if contract['mechanical_urdf_sha256'] != sha256(urdf):
            raise ValueError('BC morphology differs from the training mechanical hand')
        if (contract['effective_settings'] != scene['effective_settings']
                or contract['version'] != scene['version']
                or contract['curriculum_stage'] != scene['curriculum_stage']
                or contract['implementation_sha256'] != scene['implementation_sha256']):
            raise ValueError('BC scene differs from training scene')
        data=np.load(entry['trace'])
        expected_controller={'grasp':'grasp_joint_target_v1','forage':'forage_joint_target_v1',
                             'branch':'branch_joint_target_v1','strike':'strike_joint_target_v1'}[task]
        report=json.loads(Path(entry['metrics']).read_text())
        if report.get('controller') != expected_controller:
            raise ValueError('Demonstration action controller differs from BC target')
        x,y=data['observations_before_step'],data['submitted_actions']
        if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or not len(x):
            raise ValueError('Invalid BC observation/action layout')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (np.abs(y)>1.00001).any():
            raise ValueError('Invalid BC values')
        if not data['actions_control_fingers'].all() or not data['scene_unmodified'].all():
            raise ValueError('BC requires policy actions without scripted state overrides')
        observations.append(x);actions.append(y)
    return np.concatenate(observations),np.concatenate(actions)


def pretrain(agent, transfer, manifest_path, task, urdf, scene, output_dir, clip_obs, epochs=200):
    import torch
    x,y=load_verified_demonstrations(manifest_path,task,urdf,scene)
    observations=torch.as_tensor(x,device=agent.ppo_device)
    actions=torch.as_tensor(y,device=agent.ppo_device)
    observations=transfer.observation_for_source_policy(observations.clamp(-clip_obs,clip_obs))
    # The normalized action sign conversion is involutive.
    actions=transfer.action_for_target_hand(actions)
    optimizer=torch.optim.Adam(agent.model.parameters(),lr=1e-4)
    losses=[]
    agent.model.train()
    for epoch in range(epochs):
        order=torch.randperm(len(actions),device=actions.device)
        for ids in order.split(256):
            result=agent.model({'is_train':True,'prev_actions':actions[ids],'obs':observations[ids]})
            loss=torch.nn.functional.mse_loss(result['mus'],actions[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite BC loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()))
    report={'task':task,'samples':len(actions),'epochs':epochs,'first_loss':losses[0],
            'final_loss':losses[-1],'manifest_sha256':sha256(manifest_path),
            'input_convention':'physical_clip_then_legacy_semantic', 'mode':'verified_script_BC'}
    Path(output_dir,'behavior_cloning.json').write_text(json.dumps(report,indent=2))
    return report
