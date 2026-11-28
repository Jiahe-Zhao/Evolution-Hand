"""AX50 shared Newton world with the unchanged tray and three free pebbles.

Reuses the reviewed AX50 handle runtime (wrist PD, pair-specific palm damping and
collision-filter bookkeeping). Object geometry, mass, friction and free joints come
from outputs/nov20/ax50_pebble07/object.xml, which inspect_asset.py extracts from the
original scene. Nothing here changes object dimensions or disables global contacts.

The default command is a CPU-only static import check. `--smoke` instantiates the
GPU runtime and is a smoke step only, never an operation result.
"""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import argparse
import json
import sys

ROOT = Path((str(_EVOLUTION_ROOT)))
sys.path.insert(0, str(ROOT / 'scripts/nov20/ax50_handle13'))
# HandleRuntime already imports the reviewed wrist PD and palm-damping adapter.
from handle_runtime import HandleRuntime, newton, np, wp

OUT = ROOT / 'outputs/nov20/ax50_pebble07'
OBJECT_XML = OUT / 'object.xml'
OBJECT_BODY_NAMES = ('item', 'neighbor1', 'neighbor2')
FREE_JOINT_NAMES = ('free0', 'free1', 'free2')
SCOPE = ('shared-world import and short step only; hand is far away; '
         'not a grasp or task acceptance')


def short(label):
    return str(label).rsplit('/', 1)[-1]


class PebbleRuntime(HandleRuntime):
    """HandleRuntime with the tray and three free pebbles instead of the vessel."""

    def __init__(self, *args, object_xml=None, apply_pair_overrides=True, **kwargs):
        self.object_xml = Path(object_xml) if object_xml is not None else OBJECT_XML
        super().__init__(*args, **kwargs)
        if apply_pair_overrides:
            from pebble_pairs import install_pair_overrides
            install_pair_overrides(self)

    def _add_brush(self, b):
        if not self.object_xml.exists():
            raise FileNotFoundError(
                f'{self.object_xml} missing; run scripts/nov20/ax50_pebble07/inspect_asset.py first')

        b.default_joint_cfg.target_ke = 0.
        b.default_joint_cfg.target_kd = 0.

        from newton._src.sim.builder import _BuilderShapeCollisionFilterPairs
        if isinstance(b._shape_collision_filter_pairs, list):
            old = list(b._shape_collision_filter_pairs)
            b._shape_collision_filter_pairs = _BuilderShapeCollisionFilterPairs()
            for pair in old:
                b._shape_collision_filter_pairs.append(pair)

        b.add_mjcf(str(self.object_xml), collapse_fixed_joints=False)
        self.prop_body_count = b.body_count - self.hand_body_count

        by_name = {}
        for i, n in enumerate(b.body_label):
            name = short(n)
            if name in OBJECT_BODY_NAMES:
                by_name[name] = i
        missing = [n for n in OBJECT_BODY_NAMES if n not in by_name]
        if missing:
            raise RuntimeError(f'object.xml is missing bodies: {missing}')
        self.object_body_by_name = by_name
        self.object_bodies = [by_name[n] for n in OBJECT_BODY_NAMES]
        self.item = by_name['item']

        self.prop_dof_start = self.hand_dof_count
        self.prop_dof_count = b.joint_dof_count - self.prop_dof_start
        for d in range(self.prop_dof_start, b.joint_dof_count):
            b.joint_target_mode[d] = newton.JointTargetMode.NONE
            b.joint_target_ke[d] = 0.
            b.joint_target_kd[d] = 0.

        self.free_joint_rows = []
        for j, n in enumerate(b.joint_label):
            name = short(n)
            if name in FREE_JOINT_NAMES:
                self.free_joint_rows.append({
                    'name': name,
                    'joint': j,
                    'q_start': int(b.joint_q_start[j]),
                    'qd_start': int(b.joint_qd_start[j]),
                })
        found = tuple(r['name'] for r in self.free_joint_rows)
        if sorted(found) != sorted(FREE_JOINT_NAMES):
            raise RuntimeError(f'object.xml is missing free joints: {found}')

    def object_info(self):
        model = self.model
        return {
            'scope': 'derived runtime inspection only; not an operation result',
            'object_body_names': list(OBJECT_BODY_NAMES),
            'object_bodies': self.object_bodies,
            'body_mass_kg': [float(model.body_mass.numpy()[i]) for i in self.object_bodies],
            'free_joint_rows': self.free_joint_rows,
            'prop_dof_start': self.prop_dof_start,
            'prop_dof_count': self.prop_dof_count,
            'prop_target_ke': [float(x) for x in model.joint_target_ke.numpy()[self.prop_dof_start:]],
            'prop_target_kd': [float(x) for x in model.joint_target_kd.numpy()[self.prop_dof_start:]],
            'hand_body_count': self.hand_body_count,
            'total_body_count': int(model.body_count),
            'shape_count': int(model.shape_count),
        }


def static_check(object_xml):
    if not object_xml.exists():
        raise FileNotFoundError(
            f'{object_xml} missing; run scripts/nov20/ax50_pebble07/inspect_asset.py first')
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(object_xml), collapse_fixed_joints=False)
    model = builder.finalize(device='cpu')

    body_names = [short(n) for n in model.body_label]
    joint_names = [short(n) for n in model.joint_label]
    mass = {short(n): float(m) for n, m in zip(model.body_label, model.body_mass.numpy())}
    target_ke = [float(x) for x in model.joint_target_ke.numpy()]
    target_kd = [float(x) for x in model.joint_target_kd.numpy()]

    checks = {
        'bodies': body_names == list(OBJECT_BODY_NAMES),
        'free_joints': joint_names == list(FREE_JOINT_NAMES),
        'free_dof_18': int(model.joint_dof_count) == 18,
        'joint_q_21': int(len(model.joint_q)) == 21,
        'no_object_drive': all(x == 0.0 for x in target_ke + target_kd),
        'mass_present': all(mass.get(n, 0.0) > 0.0 for n in OBJECT_BODY_NAMES),
    }
    result = {
        'scope': 'CPU import only; runtime not instantiated; GPU smoke not run',
        'object_xml': str(object_xml),
        'body_names': body_names,
        'body_mass_kg': mass,
        'joint_names': joint_names,
        'joint_dof_count': int(model.joint_dof_count),
        'joint_q_length': int(len(model.joint_q)),
        'shape_count': int(model.shape_count),
        'checks': checks,
        'passed': all(checks.values()),
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--object-xml', type=Path, default=OBJECT_XML)
    parser.add_argument('--smoke', action='store_true',
                        help='instantiate the GPU runtime and step 0.05 s (smoke only)')
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    result = static_check(args.object_xml)
    (OUT / 'static_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        return 1
    if not args.smoke:
        return 0

    h = PebbleRuntime(side='left', wrist='six_axis', position=(.25, .25, .35),
                      gravity=(0, 0, -9.81), object_xml=args.object_xml)
    start = h.state.body_q.numpy()[h.object_bodies].copy()
    h.advance(.05, dt=.000125)
    m = h.model
    smoke = {
        'scope': SCOPE,
        'time': h.time,
        'finite': all(np.isfinite(x.numpy()).all() for x in
                      [h.state.joint_q, h.state.joint_qd, h.state.body_q,
                       h.state.particle_q, h.state.particle_qd]),
        'object_bodies': h.object_bodies,
        'object_body_names': list(OBJECT_BODY_NAMES),
        'object_mass_kg': [float(x) for x in m.body_mass.numpy()[h.object_bodies]],
        'object_start': start.tolist(),
        'object_end': h.state.body_q.numpy()[h.object_bodies].tolist(),
        'free_joint_rows': h.free_joint_rows,
        'prop_target_ke': [float(x) for x in m.joint_target_ke.numpy()[h.prop_dof_start:]],
        'prop_target_kd': [float(x) for x in m.joint_target_kd.numpy()[h.prop_dof_start:]],
        'total_dof': int(m.joint_dof_count),
        'total_q': int(len(h.state.joint_q.numpy())),
        'total_bodies': int(m.body_count),
    }
    (OUT / 'shared_smoke.json').write_text(json.dumps(smoke, indent=2), encoding='utf-8')
    print(json.dumps(smoke, indent=2))
    return 0 if smoke['finite'] else 1


if __name__ == '__main__':
    sys.exit(main())
