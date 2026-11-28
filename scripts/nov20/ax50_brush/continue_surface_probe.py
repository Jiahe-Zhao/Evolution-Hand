
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys, os, json
import numpy as np

ROOT = Path((str(_EVOLUTION_ROOT)))
P = ROOT / 'outputs/nov20/ax50_brush'
OUT = P / 'continue'
sys.path.insert(0, str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
import ax50_visualskin
sys.path.insert(0, str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
import newton
from continue_opposed_trial import prepare, R, Q, MOUNT, contacts

r = AX50WritingScene(side='left', wrist='locked', position=MOUNT, orientation=Q,
                     gravity=(0,0,-9.81), iterations=60, brush_xml=P/'tongji_scene_source.xml')
r.restore_prepared_hand(np.load(prepare()))
q = r.state.joint_q.numpy()
pq = r.state.particle_q.numpy()
body = r.state.body_q.numpy()
print('PREPARED_HAND', [(b, body[r.body_names_all.index(b)].tolist())
                        for b in r.body_names_all if b.startswith('link_')], flush=True)
print('SHAPE_LABELS', [(i, r.model.shape_label[i], r.model.shape_body.numpy()[i],
                        int(r.model.shape_type.numpy()[i]))
                       for i in range(len(r.model.shape_label)) if 'brush' not in r.model.shape_label[i]], flush=True)
print('CONTACT_ARRAYS', vars(r.rigid_observation).keys(), flush=True)
pad_bones = np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
print('PAD_BONES', np.unique(pad_bones, return_counts=True), flush=True)


def measure(x, y, z):
    q[r.brush_q_start:r.brush_q_start+7] = np.r_[R.apply([x,y,z])+MOUNT, [0,0,0,1]]
    q[r.brush_hinge_q] = 0
    r.state.joint_q.assign(q)
    r.state.joint_qd.zero_()
    newton.eval_fk(r.model, r.state.joint_q, r.state.joint_qd, r.state)
    p = R.apply([x,y,z])+MOUNT
    body = r.state.body_q.numpy()
    r.collision.reset_contact_matching()
    r.collision.collide(r.state, r.raw_contacts)
    r.rigid.update_contacts(r.rigid_observation, r.state)
    p = body[r.brush_root_body,:3]
    rep = {'local': [x,y,z], 'root': p.tolist(), 'pad': {}}
    for bone in np.unique(pad_bones):
        sel = np.flatnonzero(pad_bones==bone)+r.pad_offset
        pt = pq[sel]
        rel = pt-p
        radial = np.linalg.norm(rel[:,:2], axis=1)
        for extent in (-.005,.055):
            val = np.flatnonzero(np.abs(rel[:,2]-(p[2]-extent))<.025)
            if len(val):
                j = val[np.argmin(radial[val])]
                rep['pad'][str(bone)] = {'min_gap_mm':float((radial[j]-.005)*1000),
                                         'point':pt[j].tolist()}
    c = r.rigid_observation
    n = int(c.rigid_contact_count.numpy()[0])
    a,b = c.rigid_contact_shape0.numpy()[:n],c.rigid_contact_shape1.numpy()[:n]
    rr=[]
    for i in range(n):
        labels=[r.model.shape_label[a[i]], r.model.shape_label[b[i]]]
        if a[i]<0 or b[i]<0 or 'brush_shaft' not in labels:continue
        rr.append({'shapes':labels, 'normal':c.rigid_contact_normal.numpy()[i].tolist(),
                      'separation_mm':float(np.linalg.norm(c._rigid_contact_diff_point1_world.numpy()[i]-c._rigid_contact_diff_point0_world.numpy()[i])*1000),
                      'point0_world':c._rigid_contact_diff_point0_world.numpy()[i].tolist(),
                      'point1_world':c._rigid_contact_diff_point1_world.numpy()[i].tolist()})
    rep['rigid']=rr
    return rep

if os.getenv('MODE') == 'SCAN':
    results=[]
    for y in np.arange(.028,.067,.004):
        for z in np.arange(-.016,.025,.004):
            v=measure(.0205621569,float(y),-.0005726564+float(z))
            results.append(v)
    (OUT/'surface_scan.json').write_text(json.dumps(results,indent=2))
    print('WROTE surface_scan',flush=True)
else:
    results=[measure(.0205621569,.044,-.0005726564)]
    print(json.dumps(results,indent=2),flush=True)
