
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT/'outputs/nov20/ax50_brush'
OUT = P/'continue'
sys.path.insert(0, str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
from continue_opposed_trial import prepare, Q, MOUNT, R
import newton


def main():
    r = AX50WritingScene(side='left', wrist='locked', position=MOUNT, orientation=Q,
                         gravity=(0,0,0), iterations=60,
                         brush_xml=P/'tongji_scene_source.xml')
    r.restore_prepared_hand(np.load(prepare()))
    qbase = r.state.joint_q.numpy().copy()
    body_idx = r.model.shape_body.numpy()
    shape_x = r.model.shape_transform.numpy()
    shapes = {s: i for i,s in enumerate(r.model.shape_label)}
    pad_bones = np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
    names = ['thumb_sweep', 'digit_1_flex_1', 'digit_1_flex_2',
             'digit_3_flex_1', 'digit_3_flex_2', 'digit_3_flex_3']
    ids = [r.joint_names.index(n) for n in names]
    low = r.model.joint_limit_lower.numpy()[ids]
    high = r.model.joint_limit_upper.numpy()[ids]
    index_particles = np.flatnonzero(pad_bones == 'link_2_3') + r.pad_offset
    index_rest = r.state.particle_q.numpy()[index_particles].copy()
    index_body = r.body_names.index('link_2_3')
    prepared_body = r.state.body_q.numpy()[index_body].copy()
    rest_rotation = Rotation.from_quat(prepared_body[3:])
    index_local = rest_rotation.inv().apply(index_rest-prepared_body[:3])
    index_rigid = shapes[next(k for k in shapes if k.endswith('skin_2_3'))]

    def points(x):
        q = qbase.copy()
        q[ids] = x[:6]
        r.state.joint_q.assign(q)
        newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
        body = r.state.body_q.numpy()
        out = {}
        for name in ['skin_1_2', 'skin_3_3']:
            label = next(k for k in shapes if k.endswith(name))
            si = shapes[label]
            v = Rotation.from_quat(shape_x[si,3:]).apply(np.asarray(r.model.shape_source[si].vertices)) + shape_x[si,:3]
            v = Rotation.from_quat(body[body_idx[si],3:]).apply(v) + body[body_idx[si],:3]
            out[name] = R.inv().apply(v-MOUNT)
        current_body = body[index_body]
        index_points = Rotation.from_quat(current_body[3:]).apply(index_local) + current_body[:3]
        out['link_2_3'] = R.inv().apply(index_points-MOUNT)
        return out

    closest = []
    def residual(x):
        c = x[6:9]
        geom = points(x)
        gaps = {}
        for k,v in geom.items():
            d = v-c
            rad = np.linalg.norm(d[:,:2],axis=1)-.005
            i = np.argmin(rad)
            gaps[k] = rad[i]
            closest.append((k,v[i].tolist()))
        z = [v.mean(0)[2] for v in geom.values()]
        soft_push = np.maximum(0,np.asarray(list(gaps.values()))+0.00005)
        return soft_push*1000

    print('BOUNDS', list(zip(names,low.tolist(),high.tolist())), flush=True)
    starts = [[.65,.6,1.3,1.2,1.2,.5,.038,.044,.006]]
    rng = np.random.default_rng(7)
    for _ in range(7):
        start = np.r_[rng.uniform(np.maximum(low,0),np.minimum(high,1.5)), rng.uniform([.025,.037,-.003],[.043,.052,.012])]
        starts.append(np.clip(start, np.r_[low,[-.1,-.1,-.1]], np.r_[high,[.1,.1,.06]]))
    fits = []
    for x0 in starts:
        fit = least_squares(residual,x0,bounds=(np.r_[low,np.zeros(3)],np.r_[high,[.1,.1,.06]]),
                            diff_step=2e-4,max_nfev=140,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        row = {'cost':float(fit.cost),'x':fit.x.tolist(),'residual':residual(fit.x).tolist()}
        fits.append(row)
        print('FIT',json.dumps(row),flush=True)
    fits.sort(key=lambda a:a['cost'])
    (OUT/'tripod_fits.json').write_text(json.dumps(fits,indent=2))


if __name__ == '__main__':
    main()
