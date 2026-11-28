
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json, os, sys, time
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT / 'outputs/nov20/ax50_brush'
OUT = P / 'continue'
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_brush'))
from writing_adapter import AX50WritingScene
from tongji_controller import TongjiTrajectory
from continue_opposed_trial import R, Q, MOUNT, prepare


def point_velocity(q, qd, body, offset):
    linear = qd[body, :3]
    omega = qd[body, 3:]
    return linear + np.cross(omega, offset)


def main():
    state_path = Path(os.environ['CONTINUE_STATE'])
    seconds = float(os.getenv('SECONDS', '25'))
    dt = float(os.getenv('DT', '.0001'))
    speed = float(os.getenv('SPEED', '.04'))
    meta = json.loads((OUT / f'{state_path.stem}.json').read_text())
    locked = json.loads(meta['rows'][0].get('writing_mount', '{}'))
    pitch = float(locked.get('pitch_deg', 180.))
    rotation = Rotation.from_euler('x', pitch, degrees=True)
    mount = np.asarray(locked.get('mount', MOUNT))
    orientation = rotation.as_quat()
    source = AX50WritingScene(side='left', wrist='six_axis', position=mount,
                              orientation=orientation, gravity=(0, 0, -9.81), iterations=int(os.getenv('ITERATIONS', '100')),
                              brush_xml=P/'tongji_scene_source.xml')
    r = source
    state = np.load(state_path)
    for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd']:
        saved = state['state_' + key]
        if not np.isfinite(saved).all():
            raise ValueError(f'nonfinite saved {key}: continuation refused')
        getattr(r.state, key).assign(saved)
    newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
    r.rigid.reset(r.state, flags=0)
    r.vbd.reset(r.state, flags=0)
    r.collision.reset_contact_matching()
    for key, value in vars(r.state).items():
        dst = getattr(r.next, key, None)
        if isinstance(value, wp.array) and isinstance(dst, wp.array):
            wp.copy(dst, value)
    if 'state_target' in state:
        r.target.assign(state['state_target'])
    else:
        target = r.target.numpy()
        target[state['target_names']] = state['target']
        r.target.assign(target)
    grip_target = r.target.numpy().copy()
    trajectory = TongjiTrajectory(P/'tongji_paths.json', speed=speed)
    wrist_names = ['diagnostic_wrist_x','diagnostic_wrist_y','diagnostic_wrist_z']
    wrist_ids = [r.joint_names.index(n) for n in wrist_names]
    print('READY', trajectory.duration, grip_target[wrist_ids].tolist(), flush=True)

    initial_body = r.state.body_q.numpy().copy()
    brush_origin = initial_body[r.brush_root_body,:3].copy()
    rows, frames, particles, ink = [], [initial_body], [r.state.particle_q.numpy()], []
    wall, steps = time.time(), round(seconds/dt)
    sample_dt = float(os.getenv('SAMPLE_INTERVAL', '.005'))
    interval = max(1, round(sample_dt/dt))
    control_interval = max(1, round(.001/dt))
    stored = 0
    for k in range(1, steps+1):
        t = (k-1)*dt
        if (k-1) % control_interval == 0:
            xyz, phase, character, stroke = trajectory.sample(t)
            target = grip_target.copy()
            target[wrist_ids] += xyz
            control = r.control.joint_target.numpy()
            control[wrist_ids] = target[wrist_ids]
            r.control.joint_target.assign(control)
        r.step(dt)
        body = r.state.body_q.numpy()
        particle = r.state.particle_q.numpy()
        if not np.isfinite(body).all() or not np.isfinite(particle).all():
            raise RuntimeError(f'nonfinite simulation at {r.time}')
        if k % interval == 0 or k == steps:
            c = r.rigid_observation
            nc = int(c.rigid_contact_count.numpy()[0])
            a, b = c.rigid_contact_shape0.numpy()[:nc], c.rigid_contact_shape1.numpy()[:nc]
            labels = r.model.shape_label
            bundles = set()
            force = [0.,0.,0.]
            for ia,ib,force6 in zip(a,b,c.force.numpy()[:nc]):
                pair = (labels[ia],labels[ib])
                if 'paper' not in pair:
                    continue
                for label in pair:
                    if label.startswith('bristle_'):
                        bundles.add(int(label.split('_')[1]))
                force = (np.asarray(force)+force6[:3]).tolist()
            for bundle in bundles:
                tip = body[r.brush_tip_bodies[bundle]]
                point = tip[:3] + Rotation.from_quat(tip[3:]).apply([0.,0.,-.0044])
                if abs(point[2]-initial_body[r.brush_tip_bodies[bundle],2]) < .04:
                    ink.append([r.time,bundle,*point.tolist()])
            displacement = np.linalg.norm(body[r.brush_root_body,:3]-brush_origin)
            if phase == 'travel' and displacement > .2:
                raise RuntimeError(f'brush grip lost at {r.time}: {displacement}m displacement')
            row = {'t':r.time,'phase':phase,'character':character,'stroke':stroke,
                   'brush_q':body[r.brush_root_body].tolist(),
                   'wrist_q':r.state.joint_q.numpy()[wrist_ids].tolist(),
                   'wrist_target':target[wrist_ids].tolist(),
                   'paper_contact_bundles':sorted(bundles), 'paper_force_N':force,
                   'bristle_rms_abs_rad':float(np.sqrt(np.mean(r.state.joint_q.numpy()[r.brush_hinge_q]**2))),
                   'ink_points':len(ink)}
            rows.append(row)
            frames.append(body.copy());particles.append(particle.copy())
            print('SAMPLE',json.dumps({x:y for x,y in row.items() if x not in ['brush_q','character']}),flush=True)
            name = os.getenv('NAME','writing_progress')
            np.savez(OUT/f'{name}.npz',body_q=np.array(frames),particle_q=np.array(particles),
                     body_names=r.body_names_all,state_joint_q=r.state.joint_q.numpy(),
                     state_joint_qd=r.state.joint_qd.numpy(),state_body_q=body,
                     state_body_qd=r.state.body_qd.numpy(),state_particle_q=particle,
                     state_particle_qd=r.state.particle_qd.numpy(),state_target=r.target.numpy(),
                     ink=np.asarray(ink),brush_tip_ids=r.brush_tip_bodies)
            (OUT/f'{name}.json').write_text(json.dumps({'source_grasp':str(state_path),'dt':dt,
                     'trajectory_duration_s':trajectory.duration,'rows':rows,'status':'in progress'},ensure_ascii=False,indent=2))
    name = os.getenv('NAME','tongji_physical')
    result = {'source_grasp':str(state_path),'wall_seconds':time.time()-wall,'dt':dt,
              'text':trajectory.text,'stroke_count':15,'status':'requires independent contact-ink and grasp validation',
              'ink_is_contact_driven':True,'rows':rows}
    (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
