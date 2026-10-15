"""Explicit control contract for coordinated palm supports.

Keeps all physical DOFs. Couples four proximal flexion commands, holds support
splay neutral, and leaves thumb and distal joint commands unchanged. This is
control coordination, not a claim of a fabricated passive linkage.
"""
import math

ROOT_FLEX = tuple(f'link_0_0_to_link_{d}_0' for d in range(2, 6))
ROOT_SPLAY = tuple(f'link_{d}_mcp_spread_joint' for d in range(2, 6))
MAX_DEVIATION_RAD = math.radians(5.)


def coordinate_palm_joints(values, joint_names):
    if values.shape[-1] != len(joint_names):
        raise ValueError('Palm coordination requires the complete physical joint layout')
    ids = [joint_names.index(name) for name in ROOT_FLEX]
    spread = [joint_names.index(name) for name in ROOT_SPLAY]
    output = values.clone()
    roots = output[..., ids]
    common = roots.mean(dim=-1, keepdim=True)
    output[..., ids] = common + (roots-common).clamp(-MAX_DEVIATION_RAD, MAX_DEVIATION_RAD)
    output[..., spread] = 0.
    return output


def contract():
    return {'version':'coordinated_palm_v1','physical_joint_count':20,
            'root_flexion_joints':list(ROOT_FLEX),'neutral_splay_joints':list(ROOT_SPLAY),
            'maximum_command_deviation_from_mean_rad':MAX_DEVIATION_RAD,
            'maximum_pairwise_command_difference_rad':2*MAX_DEVIATION_RAD,
            'distal_and_thumb_commands':'unchanged',
            'mechanism':'active control coordination; passive linkage not modeled'}
