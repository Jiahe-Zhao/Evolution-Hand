#!/usr/bin/env python3
"""Run a registered Newton scene without experiment-machine paths."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def smoke(spec,side,steps):
    from portable_runtime import bootstrap
    bootstrap()
    import numpy as np
    import newton
    if spec['backend']=='ax50_soft':
        sys.path.insert(0,str(ROOT/spec['module_dir']))
        module=importlib.import_module(spec['module'])
        cls=getattr(module,spec['class'])
        kwargs={'side':side,'wrist':'six_axis','position':(.25,.25,.35),'gravity':(0,0,-9.81)}
        if spec['module']=='strike_runtime':
            kwargs.update(position=module.MOUNT,orientation=module.HAND_ROT.as_quat())
        hand=cls(**kwargs)
        initial=hand.state.body_q.numpy().copy()
        hand.advance(steps*.000125,dt=.000125)
        arrays=['joint_q','joint_qd','body_q','body_qd','particle_q','particle_qd']
        finite=all(np.isfinite(getattr(hand.state,n).numpy()).all() for n in arrays)
        assert finite,'Nonfinite physics state'
        hand.reset()
        reset_error=float(np.max(np.abs(hand.state.body_q.numpy()-initial)))
        assert reset_error<1e-6,('Reset state mismatch',reset_error)
        hand.advance(.00025,dt=.000125)
        assert all(np.isfinite(getattr(hand.state,n).numpy()).all() for n in arrays)
        return {'finite':finite,'reset_max_error':reset_error,'bodies':hand.model.body_count,'dofs':hand.model.joint_dof_count,'side':side,'steps':steps,'dt':.000125}
    builder=newton.ModelBuilder()
    builder.add_mjcf(str(ROOT/spec['xml']),collapse_fixed_joints=False)
    model=builder.finalize(device='cpu');state=model.state();next_state=model.state()
    newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,iterations=40,nconmax=8192,njmax=8192)
    control=model.control()
    for _ in range(steps):
        state.clear_forces();solver.step(state,next_state,control,None,1/480);state,next_state=next_state,state
        assert all(np.isfinite(getattr(state,n).numpy()).all() for n in ['joint_q','joint_qd','body_q','body_qd']), 'Nonfinite physics state'
    return {'finite':True,'bodies':model.body_count,'dofs':model.joint_dof_count,'steps':steps,'dt':1/480}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene',nargs='?');p.add_argument('--list',action='store_true')
    p.add_argument('--mode',choices=['smoke','demo'],default='smoke')
    p.add_argument('--side',choices=['left','right'],default='left')
    p.add_argument('--steps',type=int,help='Default: 24 rigid steps or 400 soft steps (0.05 s).')
    p.add_argument('--result',type=Path)
    a=p.parse_args();specs=json.loads((ROOT/'configs/newton_scenes.json').read_text())
    if a.list:
        for key,spec in specs.items():print(key,spec['name'],spec['backend'])
        return
    if a.scene not in specs:p.error('Select a scene listed by --list')
    spec=specs[a.scene]
    if a.steps is None:a.steps=400 if spec['backend']=='ax50_soft' else 24
    if a.steps<1:p.error('--steps must be positive')
    if a.mode=='demo':
        if 'command' not in spec:p.error('This AX50 adapter exposes --mode smoke; task-specific controllers remain research scripts.')
        env=dict(os.environ,**spec.get('env',{}));env.setdefault('MUJOCO_GL','egl')
        command=[sys.executable,str(ROOT/spec['command'][0]),*spec['command'][1:]]
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        return
    result={'scene':a.scene,'name':spec['name'],'scope':'dependency/loading/short physics smoke; not task success certification',**smoke(spec,a.side,a.steps),'passed':True}
    out=a.result or ROOT/'outputs/portable_validation'/f'{a.scene}_{a.side}.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
