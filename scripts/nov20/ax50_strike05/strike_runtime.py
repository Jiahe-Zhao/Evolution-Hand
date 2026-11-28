
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = Path((str(_EVOLUTION_ROOT)))
PKG = ROOT / 'outputs/ax50_visualskin_v3_0_0'
sys.path[:0] = [str(ROOT / 'scripts/nov20/ax50_handle13'), str(PKG)]
from handle_runtime import HandleRuntime, newton, np, wp
from scipy.spatial.transform import Rotation

OUT = ROOT / 'outputs/nov20/ax50_strike05'
ASSET = ROOT / 'outputs/nov20/hafted05/asset.xml'
MOUNT = np.array([float(os.getenv(k, v)) for k, v in [('MOUNT_X', '.25'), ('MOUNT_Y', '.25'), ('MOUNT_Z', '.35')]])
HAND_ROT = Rotation.from_matrix([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
HAMMER_ROT = HAND_ROT
GRIP_LOCAL = float(os.getenv('GRIP_LOCAL', '-.085'))
PLAN_PATH = os.getenv('GRASP_PLAN', '')
PLAN = json.loads(Path(PLAN_PATH).read_text()) if PLAN_PATH else {}
HAMMER_IN_HAND = np.array(PLAN.get('hammer_in_hand', [float(os.getenv('HAND_X', '0')), float(os.getenv('HAND_Y', '-.095')), float(os.getenv('HAND_Z', '-.01'))]), dtype=float)
TARGET = np.array([float(os.getenv(k, v)) for k, v in [('TX', '.50'), ('TY', '.10'), ('TZ', '.20')]])

FINGER_JOINTS = {
    finger: [f'arch_{digit}', f'digit_{digit}_flex_1', f'digit_{digit}_flex_2', f'digit_{digit}_flex_3']
    for digit, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'little'], 2)
}
THUMB_JOINTS = ['thumb_sweep', 'arch_1', 'thumb_axial', 'digit_1_flex_1', 'digit_1_flex_2']
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little']


def hand_targets(action_names, fingers, thumb, wrist=None):
    target = np.zeros(len(action_names))
    values = dict(zip([name for names in FINGER_JOINTS.values() for name in names], np.asarray(fingers).ravel()))
    values.update(zip(THUMB_JOINTS, np.asarray(thumb).ravel()))
    if wrist is not None:
        values.update(zip(['diagnostic_wrist_x', 'diagnostic_wrist_y', 'diagnostic_wrist_z'], wrist))
    for name, value in values.items():
        if name not in action_names:
            raise KeyError(f'missing named action {name}')
        target[action_names.index(name)] = value
    return target


def evaluate_contacts(h, recs):
    finger_n = dict.fromkeys(FINGER_NAMES, 0.)
    parts = dict.fromkeys(['stone', 'wood', 'stone_target', 'wood_target', 'hand_target'], 0.)
    palm = support = rigid_penetration = soft_indentation = 0.
    links = set()
    finite = True
    for r in recs:
        finite = finite and math.isfinite(r['fn']) and math.isfinite(r['depth'])
        fn = r['fn'] if math.isfinite(r['fn']) else 1.e30
        if r['kind'] == 'rigid':
            rigid_penetration = max(rigid_penetration, r['depth'] if math.isfinite(r['depth']) else 1.)
        else:
            soft_indentation = max(soft_indentation, r['depth'] if math.isfinite(r['depth']) else 1.)
        if not math.isfinite(r['fn']) or fn <= 1.e-9:
            continue
        if r['object'] in parts:
            parts[r['object']] += fn
        if r['finger'] and r['object'] in ['stone', 'wood', 'hammer']:
            finger_n[r['finger']] += fn
        if r['palm'] and r['object'] in ['stone', 'wood', 'hammer']:
            palm += fn
        if r['support']:
            support += fn
        if r['hand'] and not r['support']:
            links.add(r['hand'])
    return dict(finger_n=finger_n, parts=parts, palm=palm, support=support,
                rigid_penetration=rigid_penetration, soft_indentation=soft_indentation,
                finite=finite, links=links, active=[f for f in FINGER_NAMES if finger_n[f] > .01],
                hand_total=parts['stone'] + parts['wood'])


def relative_hammer(h, bq):
    hammer = bq[h.hammer_body]
    palm = bq[h.body_names.index('link_2_0')]
    rotation = Rotation.from_quat(palm[3:])
    return hammer, rotation.inv().apply(hammer[:3] - palm[:3]), rotation.inv() * Rotation.from_quat(hammer[3:])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare_scene():
    OUT.mkdir(parents=True, exist_ok=True)
    root = _evo_parse_xml(ASSET).getroot()
    body = root.find("worldbody/body[@name='hafted_hammer']")
    collision = [dict(g.attrib) for g in body.findall('geom') if g.get('contype') != '0']
    assert len(collision) == 20
    assert all(g['friction'] == '.8 .004 .0001' for g in collision)
    assert abs(sum(float(geom.get('mass')) for geom in body.findall('geom')) - .30243683621108713) < 1.e-12
    original_mass = sum(float(g['mass']) for g in collision)
    assert abs(original_mass - .30243483621108713) < 1.e-12
    position = MOUNT + HAND_ROT.apply(HAMMER_IN_HAND) - HAMMER_ROT.apply([0, 0, GRIP_LOCAL])
    body.set('pos', ' '.join(map(str, position)))
    body.set('quat', ' '.join(map(str, HAMMER_ROT.as_quat()[[3, 0, 1, 2]])))
    world = root.find('worldbody')
    fmt = lambda a: ' '.join(map(str, a))
    if os.getenv('BARE', '0') != '1':
        ET.SubElement(world, 'geom', name='target', type='ellipsoid', pos=fmt(TARGET), size='.013 .013 .012', rgba='.67 .36 .15 1', contype='2', conaffinity='3')
    ET.SubElement(world, 'geom', name='anvil', type='cylinder', pos=fmt(TARGET - [0, 0, .018]), size='.014 .006', rgba='.28 .30 .29 1', contype='2', conaffinity='3')
    path = OUT / ('scene_bare.xml' if os.getenv('BARE') == '1' else 'scene.xml')
    ET.ElementTree(root).write(path)
    assert [dict(g.attrib) for g in body.findall('geom') if g.get('contype') != '0'] == collision
    return path, position, original_mass

class StrikeRuntime(HandleRuntime):
    def _add_brush(self, b):
        self.scene_path, self.initial_hammer_position, self.original_mass = prepare_scene()
        b.default_joint_cfg.target_ke = 0.
        b.default_joint_cfg.target_kd = 0.
        from newton._src.sim.builder import _BuilderShapeCollisionFilterPairs
        if isinstance(b._shape_collision_filter_pairs, list):
            old = list(b._shape_collision_filter_pairs)
            b._shape_collision_filter_pairs = _BuilderShapeCollisionFilterPairs()
            for pair in old:
                b._shape_collision_filter_pairs.append(pair)
        hand_shapes = b.shape_count
        b.add_mjcf(str(self.scene_path), collapse_fixed_joints=False)
        self.hand_shapes = np.arange(hand_shapes, dtype=int)
        labels = [str(n).rsplit('/', 1)[-1] for n in b.shape_label]
        self.hammer_body = next(i for i, n in enumerate(b.body_label) if str(n).endswith('/hafted_hammer'))
        self.object_bodies = [self.hammer_body]
        shape_body = np.asarray(b.shape_body)
        self.hammer_shapes = np.array([i for i, n in enumerate(labels) if n.startswith(('stone_band', 'wood_grip'))], dtype=int)
        self.stone_shapes = np.array([i for i, n in enumerate(labels) if n.startswith('stone_band')], dtype=int)
        self.wood_shapes = np.array([i for i, n in enumerate(labels) if n.startswith('wood_grip')], dtype=int)
        self.target_shapes = np.array([i for i, n in enumerate(labels) if n == 'target'], dtype=int)
        self.support_shapes = np.array([i for i, n in enumerate(labels) if n in ('floor', 'anvil', 'target')], dtype=int)
        assert len(self.hammer_shapes) == 20 and len(self.stone_shapes) == 11 and len(self.wood_shapes) == 9
        assert (shape_body[self.hammer_shapes] == self.hammer_body).all()
        self.prop_dof_start = self.hand_dof_count
        self.hammer_free_dofs = np.arange(self.hand_dof_count, b.joint_dof_count, dtype=int)
        for d in self.hammer_free_dofs:
            b.joint_target_mode[d] = newton.JointTargetMode.NONE
            b.joint_target_ke[d] = 0.
            b.joint_target_kd[d] = 0.
        assert len(self.hammer_free_dofs) == 6
        self.prop_body_count = 1
        self.contact_distance = None

    def _build(self, *args):
        super()._build(*args)
        self.contact_distance = wp.zeros(self.rigid_observation.rigid_contact_max)
        mass = float(self.model.body_mass.numpy()[self.hammer_body])
        assert abs(mass - self.original_mass) < 2.1e-6, (mass, self.original_mass)
        assert (self.model.joint_target_ke.numpy()[self.hammer_free_dofs] == 0).all()
        assert (self.model.joint_target_kd.numpy()[self.hammer_free_dofs] == 0).all()

    def sample_contact(self):
        c = self.rigid_observation
        sb = self.model.shape_body.numpy()
        newton.eval_rigid_contact_kinematics(self.model, self.state, c, out_distance=self.contact_distance)
        n = int(c.rigid_contact_count.numpy()[0])
        if n:
            s0 = c.rigid_contact_shape0.numpy()[:n]
            s1 = c.rigid_contact_shape1.numpy()[:n]
            sb = self.model.shape_body.numpy()
            a, b = sb[s0], sb[s1]
            ff = c.force.numpy()[:n, :3]
            nn = c.rigid_contact_normal.numpy()[:n]
            dd = self.contact_distance.numpy()[:n]
        else:
            s0 = s1 = a = b = np.array([], dtype=int)
            ff = nn = np.empty((0, 3)); dd = np.empty(0)
        records = []
        for own, other, so, ss, sign in [(a, b, s0, s1, 1), (b, a, s1, s0, -1)]:
            for j in np.flatnonzero((own >= 0) & (own < self.hand_body_count) & (other == self.hammer_body)):
                fn = abs(float(ff[j] @ nn[j]))
                stone = ss[j] in self.stone_shapes
                hand = self.body_names[own[j]]
                records.append(dict(kind='rigid', hand=hand, particle=-1, object='stone' if stone else 'wood', fn=fn,
                                    depth=max(0., -float(dd[j])), target=False, support=False,
                                    finger=self._finger(hand), palm=hand == 'link_2_0'))
            for j in np.flatnonzero((own == self.hammer_body) & (other >= self.hand_body_count)):
                fn = abs(float(ff[j] @ nn[j]))
                target = ss[j] in self.target_shapes
                stone = so[j] in self.stone_shapes
                records.append(dict(kind='rigid', hand='fixed_target' if target else 'external_support', particle=-1,
                                    object=('stone' if stone else 'wood') + ('_target' if target else '_support'),
                                    fn=fn, depth=max(0., -float(dd[j])), target=target, support=True, finger='', palm=False))
            for j in np.flatnonzero((own >= 0) & (own < self.hand_body_count) & np.isin(ss, self.target_shapes)):
                hand = self.body_names[own[j]]
                records.append(dict(kind='rigid', hand=hand, particle=-1, object='hand_target',
                                    fn=abs(float(ff[j] @ nn[j])), depth=max(0., -float(dd[j])), target=True, support=False,
                                    finger=self._finger(hand), palm=False))
        c = self.contacts
        count = int(c.soft_contact_count.numpy()[0])
        if count:
            body = sb[c.soft_contact_shape.numpy()[:count]]
            ff = self.out_f.numpy()[:count]
            nn = c.soft_contact_normal.numpy()[:count]
            ii = c.soft_contact_indices.numpy()[:count]
            bb = c.soft_contact_barycentric.numpy()[:count]
            depths = self.out_depth.numpy()[:count]
            owners = self.attachment_owner.numpy()
            for j in np.flatnonzero(body == self.hammer_body):
                pid = int(ii[j, np.argmax(bb[j])])
                owner = int(owners[pid]) if 0 <= pid < len(owners) else -1
                hand = self.body_names[owner] if owner >= 0 else 'palm_soft' if 0 <= pid < self.palm_count else 'unmapped_soft_particle'

                records.append(dict(kind='soft', hand=hand, particle=pid, object='hammer', fn=abs(float(ff[j] @ nn[j])),
                                    depth=max(0., float(depths[j])), target=False, support=False,
                                    finger=self._finger(hand), palm=0 <= pid < self.palm_count))
        return records

    @staticmethod
    def _finger(name):
        for number, finger in enumerate(['thumb', 'index', 'middle', 'ring', 'little'], 1):
            if f'digit_{number}' in name or f'link_{number}_' in name:
                return finger
        return ''

    def get_checkpoint(self):
        return {key: getattr(self.state, key).numpy() for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd']} | dict(feedback=self.feedback.numpy(), target=self.target.numpy(), time=self.time)


def construct(side='left'):
    return StrikeRuntime(side=side, wrist='six_axis', position=MOUNT, orientation=HAND_ROT.as_quat(),
                         gravity=(0, 0, -9.81), device=os.getenv('SIM_DEVICE', 'cuda:0'))

def static_check(h):
    names = [str(x).rsplit('/', 1)[-1] for x in h.model.shape_label]
    return dict(device=h.device, asset=str(ASSET), asset_sha256=sha256(ASSET), scene=str(h.scene_path), scene_sha256=sha256(h.scene_path),
                collision_geom_count=len(h.hammer_shapes), stone_geom_count=len(h.stone_shapes), wood_geom_count=len(h.wood_shapes),
                original_collision_mass_kg=h.original_mass, imported_body_mass_kg=float(h.model.body_mass.numpy()[h.hammer_body]),
                geom_friction_xml='.8 .004 .0001 (all 20 collision geoms; pair contact coefficients are neither generated nor raised)',
                free_dofs=h.hammer_free_dofs.tolist(), free_joint_target_modes=h.model.joint_target_mode.numpy()[h.hammer_free_dofs].tolist(),
                free_joint_ke=h.model.joint_target_ke.numpy()[h.hammer_free_dofs].tolist(), free_joint_kd=h.model.joint_target_kd.numpy()[h.hammer_free_dofs].tolist(),
                hammer_initial=h.state.body_q.numpy()[h.hammer_body].tolist(), target=TARGET.tolist(), mount=MOUNT.tolist(),
                hand_orientation_xyzw=HAND_ROT.as_quat().tolist(), hand_xyzw=HAND_ROT.as_quat().tolist(),
                hammer_in_hand=HAMMER_IN_HAND.tolist(), grip_local_z=GRIP_LOCAL, target_size_m=[.013, .013, .012], anvil_size_m=[.014, .006],
                action_names=h.action_names, action_indices=h.action_indices.tolist(), action_lower=h.action_lower.tolist(), action_upper=h.action_upper.tolist(),
                shape_names=names, body_names=h.body_names_all, self_collision=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    h = construct()
    (OUT / 'static_check.json').write_text(json.dumps(static_check(h), indent=2))
    if args.smoke:
        h.advance(.05, dt=.000125)
        result = dict(scope='actual Newton 0.05 s initialization smoke; not grasp or strike acceptance', time_s=h.time,
                      hammer_position=h.state.body_q.numpy()[h.hammer_body].tolist(),
                      finite=all(np.isfinite(getattr(h.state, key).numpy()).all() for key in ['joint_q', 'joint_qd', 'body_q', 'body_qd', 'particle_q', 'particle_qd']),
                      contact_count=len(h.sample_contact()))
        (OUT / 'shared_smoke.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
