"""Scene-local MuJoCo pair parameters for the AX50 pebble runtime."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
from types import MethodType
import re

from pebble_runtime import np, wp
from mujoco_warp._src import types as mjw_types


LEGACY_XML = Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/pebble.xml'))


def short(name):
    return str(name).rsplit('/', 1)[-1]


def key(name):
    return re.sub(r'_\d+$', '', short(name))


@wp.kernel
def apply_pair_parameters(
    nacon: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    friction: wp.array(dtype=mjw_types.vec5f),
    solref: wp.array(dtype=wp.vec2),
    solimp: wp.array(dtype=mjw_types.vec5f),
    dim: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    pair_friction: wp.array(dtype=mjw_types.vec5f),
    pair_solref: wp.array(dtype=wp.vec2),
    pair_solimp: wp.array(dtype=mjw_types.vec5f),
    pair_dim: wp.array(dtype=wp.int32),
    ngeom: int,
    enabled: int,
):
    i = wp.tid()
    if enabled == 0 or i >= nacon[0]:
        return
    g0 = geom[i][0]
    g1 = geom[i][1]
    index = g0 * ngeom + g1
    if active[index] != 0:
        friction[i] = pair_friction[index]
        solref[i] = pair_solref[index]
        solimp[i] = pair_solimp[index]
        dim[i] = pair_dim[index]


def _pair_values(model, i):
    return {
        'friction': np.asarray(model.pair_friction[i], dtype=np.float32),
        'solref': np.asarray(model.pair_solref[i], dtype=np.float32),
        'solimp': np.asarray(model.pair_solimp[i], dtype=np.float32),
        'condim': int(model.pair_dim[i]),
    }


def install_pair_overrides(runtime):
    solver = runtime.rigid
    if solver._use_mujoco_contacts:
        raise RuntimeError('scene07 pair overrides require use_mujoco_contacts=False')
    model = solver.mj_model
    ngeom = int(model.ngeom)
    geoms = {key(model.geom(i).name): i for i in range(ngeom)}
    active = np.zeros((ngeom, ngeom), dtype=np.int32)
    dims = np.zeros((ngeom, ngeom), dtype=np.int32)
    friction = np.zeros((ngeom, ngeom, 5), dtype=np.float32)
    solref = np.zeros((ngeom, ngeom, 2), dtype=np.float32)
    solimp = np.zeros((ngeom, ngeom, 5), dtype=np.float32)

    def put(a, b, values):
        ga = a if isinstance(a, int) else geoms[a]
        gb = b if isinstance(b, int) else geoms[b]
        for g0, g1 in ((ga, gb), (gb, ga)):
            active[g0, g1] = 1
            dims[g0, g1] = values['condim']
            friction[g0, g1] = values['friction']
            solref[g0, g1] = values['solref']
            solimp[g0, g1] = values['solimp']

    object_pairs = []
    for i in range(int(model.npair)):
        a, b = key(model.geom(model.pair_geom1[i]).name), key(model.geom(model.pair_geom2[i]).name)
        put(a, b, _pair_values(model, i))
        object_pairs.append((a, b))
    if len(object_pairs) != 21:
        raise RuntimeError(f'expected 21 original object pairs, got {len(object_pairs)}')

    mujoco = solver._mujoco
    old = mujoco.MjModel.from_xml_path(str(LEGACY_XML))
    hand_pairs = []
    for i in range(int(old.npair)):
        a, b = key(old.geom(old.pair_geom1[i]).name), key(old.geom(old.pair_geom2[i]).name)
        if a.startswith('handgeom') and b in ('stone0', 'foot0'):
            hand_pairs.append((a, b, _pair_values(old, i)))
    if len(hand_pairs) != 204:
        raise RuntimeError(f'expected 204 legacy hand pairs, got {len(hand_pairs)}')
    signature = {
        (tuple(v['friction']), tuple(v['solref']), tuple(v['solimp']), v['condim'])
        for _, _, v in hand_pairs
    }
    if len(signature) != 1:
        raise RuntimeError(f'legacy hand pair parameters are not uniform: {len(signature)} variants')
    hand_values = hand_pairs[0][2]

    shape_body = runtime.model.shape_body.numpy()
    hand_shape_count = sum(
        0 <= int(body) < runtime.hand_body_count and not str(label).endswith('_visual')
        for body, label in zip(shape_body, runtime.model.shape_label))
    shape_to_geom = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
    hand_geoms = []
    for geom, shape in enumerate(shape_to_geom):
        if shape >= 0 and 0 <= int(shape_body[int(shape)]) < runtime.hand_body_count:
            hand_geoms.append(int(geom))
    for geom in hand_geoms:
        for target in ('stone0', 'foot0'):
            put(geom, target, hand_values)

    contact = solver.mjw_data.contact
    arrays = {
        name: wp.array(value.reshape((-1, value.shape[-1])) if value.ndim == 3 else value.reshape(-1), dtype=getattr(contact, contact_name).dtype, device=runtime.model.device)
        for name, value, contact_name in (
            ('active', active, 'dim'),
            ('dim', dims, 'dim'),
            ('friction', friction, 'friction'),
            ('solref', solref, 'solref'),
            ('solimp', solimp, 'solimp'),
        )
    }
    original = solver._convert_contacts_to_mjwarp

    def convert_with_pairs(self, newton_model, state_in, contacts):
        original(newton_model, state_in, contacts)
        wp.launch(
            apply_pair_parameters,
            dim=self.mjw_data.naconmax,
            inputs=[
                self.mjw_data.nacon, self.mjw_data.contact.geom,
                self.mjw_data.contact.friction, self.mjw_data.contact.solref,
                self.mjw_data.contact.solimp, self.mjw_data.contact.dim,
                arrays['active'], arrays['friction'], arrays['solref'],
                arrays['solimp'], arrays['dim'], ngeom,
                int(runtime.pair_overrides_enabled),
            ],
            device=newton_model.device,
        )

    runtime.pair_overrides_enabled = True
    runtime.pair_override_arrays = arrays
    runtime.pair_override_info = {
        'scope': 'scene07 only; instance method hook; no global shape material changes',
        'legacy_xml': str(LEGACY_XML),
        'legacy_npair': int(old.npair),
        'legacy_hand_pairs': len(hand_pairs),
        'object_pairs': len(object_pairs),
        'ax50_rigid_shapes': hand_shape_count,
        'ax50_mujoco_hand_geoms': len(hand_geoms),
        'hand_shapes_without_mujoco_geom': hand_shape_count - len(hand_geoms),
        'mapped_hand_object_rules': len(hand_geoms) * 2,
        'active_rule_pairs': len(object_pairs) + len(hand_geoms) * 2,
        'hand_values': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in hand_values.items()},
        'soft_pads': 'unchanged soft-contact physics; not equivalent to the legacy rigid hand',
    }
    solver._convert_contacts_to_mjwarp = MethodType(convert_with_pairs, solver)
