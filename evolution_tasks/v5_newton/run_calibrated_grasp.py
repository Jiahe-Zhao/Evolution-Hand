"""Executable Newton four-task interface validation entry point."""
import argparse,json,time
from pathlib import Path
from isaaclab.app import add_launcher_args,launch_simulation
p=argparse.ArgumentParser();p.add_argument('--task',choices=['grasp','branch','forage','strike'],required=True)
p.add_argument('--asset',required=True);p.add_argument('--urdf',required=True);p.add_argument('--output',required=True)
p.add_argument('--control-hz',type=int,choices=[60,120,240,480],default=60)
p.add_argument('--num_envs',type=int,default=4);p.add_argument('--steps',type=int,default=0);p.add_argument('--seed',type=int,default=31)
p.add_argument('--ppo-iterations',type=int,default=0)
p.add_argument('--fingertip-grid',action='store_true')
p.add_argument('--wrist-test',action='store_true')
p.add_argument('--wrist-free-space',action='store_true')
p.add_argument('--skin-observation',action='store_true',help='Append 20 named metacarpal/digit state and tactile regions')
add_launcher_args(p);a=p.parse_args()
import torch
import isaaclab.sim as sim
from isaaclab.utils import instantiate
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg
from task_scene import create_scene_config
from task_env import TaskEnv
root=Path(__file__).resolve().parent;config=json.loads((root/'configs'/f'{a.task}.json').read_text())
config["paired_v5"]=True
config["left_pad"]=Path(a.asset).parent.name=="left"
config["collision_asset"]=str(Path(a.asset).resolve())
config['skin_observation']=True
config['fingertip_grid']=a.fingertip_grid
config['physics_dt']=1/480;config['decimation']=int(480/a.control_hz);config['control_dt']=1/a.control_hz
config['ppo']['num_envs']=a.num_envs
config['ppo']['long_training_steps']=a.ppo_iterations*a.num_envs*config['ppo']['rollout_steps']
config['contact_profile']='wrap_calibrated_v1_10000_100_force_space'
import hashlib
config['asset_sha256']=hashlib.sha256(Path(a.asset).read_bytes()).hexdigest()
config['signature']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
import hashlib,shutil
snapshot=out/'source_snapshot';snapshot.mkdir()
source_hashes={}
for filename in ['run_calibrated_grasp.py','calibrated_materials.py','task_scene.py','task_env.py','task_metrics.py','frame_math.py','tactile_adapter.py','controller_config.py','reset_clearance.py','task_materials.py','short_ppo.py','evaluate_policy.py','skin_observation.py','fingertip_contact_grid.py','tactile_slip.py']:
    source=root/filename;shutil.copy2(source,snapshot/filename);source_hashes[filename]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
(out/'effective_config.json').write_text(json.dumps(config,indent=2))
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
scfg=sim.SimulationCfg(dt=config['physics_dt'],device=a.device,physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False,integrator='implicitfast',cone='elliptic',impratio=10.,njmax=4096,nconmax=2048),collision_cfg=NewtonCollisionPipelineCfg()))
with launch_simulation(scfg,a):
    ctx=sim.SimulationContext(scfg)
    cfg,rows,objects,targets=create_scene_config(config,a.asset,a.urdf,a.num_envs)
    from isaaclab_newton.cloner import newton_builder_world_hook
    from calibrated_materials import builder_hook,verify_backend
    with newton_builder_world_hook(builder_hook(a.task)):
        scene=instantiate(cfg)
    from calibrated_materials import apply
    (out/'material_contract.json').write_text(json.dumps(apply(ctx.stage,a.task),indent=2))
    ctx.reset();scene.update(0.)
    from isaaclab_newton.physics import NewtonManager
    (out/'backend_materials.json').write_text(json.dumps(verify_backend(NewtonManager.get_model(),a.task),indent=2))
    env=TaskEnv(ctx,scene,config,rows,objects,targets,a.seed,a.urdf)
    if any(__import__("os").environ.get(k)=="1" for k in ("V5_CONTACT_AB","V5_WRIST_AB")):
        from contact_ab import run as contact_probe
        (out/"contact_ab.json").write_text(json.dumps(contact_probe(env),indent=2))
        env.reset()
    if __import__("os").environ.get("V5_WRIST_AB")=="1":
        from probe_pair import wrist
        (out/"wrist_ab.json").write_text(json.dumps(wrist(env)))
    import numpy as np
    model=NewtonManager.get_model()
    masses=model.body_mass.numpy();inertias=model.body_inertia.numpy()
    eig=np.linalg.eigvalsh(inertias);dynamic=masses>0
    valid=np.isfinite(eig).all(1)&(eig[:,0]>0)&(eig[:,2]<=eig[:,:2].sum(1)+1e-10)
    assert valid[dynamic].all(), 'Nonphysical backend inertia'
    (out/'backend_inertia_audit.json').write_text(json.dumps(dict(dynamic_bodies=int(dynamic.sum()),passed=True,rows=[dict(label=str(label),mass_kg=float(masses[i]),inertia_eigenvalues=eig[i].tolist()) for i,label in enumerate(model.body_label)]),indent=2))
    (out/'interface_contract.json').write_text(json.dumps(dict(signature=env.signature,**env.contract),indent=2))
    if a.wrist_free_space:
        for name in env.object_names:env.initial[name][:,0]+=2.
        env.reset()
        env.compute_reward=lambda: (torch.zeros(env.num_envs,device=env.device),torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),torch.zeros(env.num_envs,device=env.device,dtype=torch.bool))
    obs=env.observe();steps=a.steps or env.max_steps+2
    initial=env.target[:,env.fids].clone();limits=env.limits[env.fids]
    neutral=2*(initial-limits[:,0])/(limits[:,1]-limits[:,0])-1
    action=torch.zeros(env.num_envs,env.action_size,device=env.device);action[:,:len(env.fids)]=neutral
    report=dict(task=a.task,steps=0,finite=True,episodes=0,successes=0,max_joint_speed=0.,max_limit_violation=0.,full_episode_length=env.max_steps,
        scope='interface rollout with small bounded perturbations; not learned success or backend parity')
    report['free_space_wrist_fixture']=a.wrist_free_space
    started=time.monotonic();motion=[]
    for step in range(steps):
        import math
        candidate=action.clone();candidate[:,:len(env.fids)]+=.01*math.sin(step*.02)
        if a.wrist_test:
            candidate[:,-3:]=env.tensor([.10*math.sin(step*.03),.10*math.sin(step*.025),.10*math.sin(step*.02)])
        obs,reward,term,trunc,info=env.step(candidate)
        if env.rids:
            report['max_wrist_rotation_rad']=max(report.get('max_wrist_rotation_rad',0.),float(env.hand.data.joint_pos.torch[:,env.rids].abs().max()))
            active=~(term|trunc)
            if active.any():
                tracking=(env.hand.data.joint_pos.torch[active][:,env.rids]-env.target[active][:,env.rids]).abs()
                report['max_wrist_tracking_error_rad']=max(report.get('max_wrist_tracking_error_rad',0.),float(tracking.max()))
                if a.wrist_free_space:assert tracking.max()<.1, 'Free-space wrist tracking error exceeds 0.1 rad'
            report['wrist_angles_rad']=env.hand.data.joint_pos.torch[0,env.rids].cpu().tolist()
        if env.fingertip_grid is not None:
            g=env.fingertip_grid
            report['tactile_contact_samples']=report.get('tactile_contact_samples',0)+int(g.contact_count.sum())
            report['max_tactile_force_N']=max(report.get('max_tactile_force_N',0.),float(g.total_force.max()))
            report['slip_samples']=report.get('slip_samples',0)+int(g.slip.sum())
            report['max_grid_force_error_N']=max(report.get('max_grid_force_error_N',0.),float((g.grid.sum((-1,-2))+g.dropped_force-g.total_force).abs().max()))
            report['out_of_field_force_sum_N']=report.get('out_of_field_force_sum_N',0.)+float(g.dropped_force.sum())
        assert obs.shape==(env.num_envs,env.observation_size)
        assert all(torch.isfinite(x).all() for x in [obs,reward,env.hand.data.joint_pos.torch,env.hand.data.joint_vel.torch])
        report['timeouts']=report.get('timeouts',0)+int(trunc.sum())
        report['episodes']+=int((term|trunc).sum());report['successes']+=int(info['episode_success'].sum())
        report['max_joint_speed']=max(report['max_joint_speed'],float(env.hand.data.joint_vel.torch.abs().max()))
        q=env.hand.data.joint_pos.torch
        report['max_limit_violation']=max(report['max_limit_violation'],float(torch.maximum(env.limits[:,0]-q,q-env.limits[:,1]).clamp_min(0).max()))
        report['steps']=step+1
        if step%6==0:
            motion.append(dict(time=step*env.dt,hand=env.hand.data.body_link_pose_w.torch[0].cpu().tolist(),objects={name:env.pose(name)[0].cpu().tolist() for name in env.object_names}))
        if step%120==0:
            (out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if env.num_envs>1:
        before={name:env.pose(name).clone() for name in env.object_names};before_q=env.hand.data.joint_pos.torch.clone();before_age=env.age.clone()
        if env.fingertip_grid is not None:
            grid_before=env.fingertip_grid.grid.clone();slip_before=env.fingertip_grid.slip.clone()
        env.reset([0])
        if env.fingertip_grid is not None:
            assert not env.fingertip_grid.grid[0].any() and not env.fingertip_grid.slip[0].any()
            assert torch.equal(env.fingertip_grid.grid[1:],grid_before[1:])
            assert torch.equal(env.fingertip_grid.slip[1:],slip_before[1:])
            report['tactile_partial_reset_passed']=True
        assert torch.equal(env.hand.data.joint_pos.torch[1:],before_q[1:])
        assert torch.equal(env.age[1:],before_age[1:]) and env.age[0]==0
        for name in before:assert torch.equal(env.pose(name)[1:],before[name][1:]),name
        assert not env.seen[0].any() and env.streak[0]==0
        assert torch.equal(env.best[0],env.progress[0])
        report['partial_reset_passed']=True
    report.update(elapsed_s=time.monotonic()-started,completed=True,ran_at_least_episode_length=steps>=env.max_steps,full_episode_exercised=report.get('timeouts',0)>0)
    from dynamic_surface import DynamicSurface
    from pxr import Usd
    binding=Path(a.asset).with_name("visual_binding.json")
    surface=DynamicSurface(binding)
    # Render on a separate stage: visual skinning must never modify collisions.
    preview=Usd.Stage.Open(a.asset).Flatten()
    stage=Usd.Stage.Open(preview)
    poses=dict(zip(env.hand.body_names,motion[-1]["hand"]))
    surface.update_stage(stage,poses)
    stage.GetRootLayer().Export(str(out/"deformed_surface.usda"))
    report["visual_binding_sha256"]=hashlib.sha256(binding.read_bytes()).hexdigest()
    (out/'motion.json').write_text(json.dumps(dict(body_names=env.hand.body_names,frames=motion)))
    if a.ppo_iterations:
        from short_ppo import run
        report['ppo']=run(env,out,a.ppo_iterations)
        from evaluate_policy import evaluate
        report['evaluation']=[evaluate(env,out/'short_ppo.pth',mode=mode) for mode in ['mean','sample']]
        (out/'evaluation.json').write_text(json.dumps(report['evaluation'],indent=2))
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
