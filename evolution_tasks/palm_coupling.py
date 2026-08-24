"""Action-space constraints that approximate a palm arch for the evolved hand.

The model has no explicit metacarpal/palm mesh.  Applying the constraint before
joint targets are scaled gives the four long fingers a shared virtual palm:
neighbouring bases cannot diverge sharply, while the thumb remains free to
oppose them.
"""

from __future__ import annotations

import torch


# Action ordering is shared by the five-finger Evolution hand tasks:
# thumb (3 DoFs), then four long fingers (4 DoFs each).
_BASE = (3, 7, 11, 15)
_PROXIMAL = (4, 8, 12, 16)
_MAX_NEIGHBOUR_BASE_GAP = 0.45
_FINGER_GROUPS = ((0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10), (11, 12, 13, 14), (15, 16, 17, 18))
_LONG_FINGER_GROUPS = _FINGER_GROUPS[1:]


def apply_virtual_palm_coupling(actions: torch.Tensor) -> torch.Tensor:
    """Return normalized actions with coupled long-finger roots.

    Central fingers lead the arch.  A mild proximal coupling keeps a grasping
    sweep coherent without forcing identical finger flexion.
    """
    if actions.shape[-1] != 19:
        raise ValueError("Virtual palm coupling expects the 19-DoF Evolution hand action layout.")

    coupled = actions.clone()
    bases = actions[:, _BASE]
    smoothed = torch.empty_like(bases)
    smoothed[:, 0] = 0.78 * bases[:, 0] + 0.22 * bases[:, 1]
    smoothed[:, 1] = 0.16 * bases[:, 0] + 0.68 * bases[:, 1] + 0.16 * bases[:, 2]
    smoothed[:, 2] = 0.16 * bases[:, 1] + 0.68 * bases[:, 2] + 0.16 * bases[:, 3]
    smoothed[:, 3] = 0.22 * bases[:, 2] + 0.78 * bases[:, 3]

    # A transverse palm limits adjacent metacarpal separation.
    constrained = torch.empty_like(smoothed)
    constrained[:, 0] = smoothed[:, 0]
    for index in range(1, 4):
        constrained[:, index] = torch.clamp(
            smoothed[:, index],
            constrained[:, index - 1] - _MAX_NEIGHBOUR_BASE_GAP,
            constrained[:, index - 1] + _MAX_NEIGHBOUR_BASE_GAP,
        )
    coupled[:, _BASE] = constrained

    proximal = actions[:, _PROXIMAL]
    coupled[:, _PROXIMAL] = 0.88 * proximal + 0.12 * constrained
    return coupled


def apply_branch_finger_coordination(actions: torch.Tensor, max_deviation: float) -> torch.Tensor:
    """Keep the four long fingers within a shared flexion-amplitude band.

    The thumb remains independent for opposition.  The four long fingers are
    projected into a bounded interval around their shared mean, preventing two
    digits from remaining nearly static while the other two close strongly.
    """
    if actions.shape[-1] != 19:
        raise ValueError("Branch finger coordination expects the 19-DoF Evolution hand action layout.")

    coordinated = actions.clone()
    digit_scores = torch.stack([actions[:, group].mean(dim=-1) for group in _LONG_FINGER_GROUPS], dim=-1)
    shared_score = digit_scores.mean(dim=-1, keepdim=True)
    bounded_scores = torch.clamp(digit_scores, shared_score - max_deviation, shared_score + max_deviation)
    for digit_index, group in enumerate(_LONG_FINGER_GROUPS):
        score_offset = (bounded_scores[:, digit_index] - digit_scores[:, digit_index]).unsqueeze(-1)
        coordinated[:, group] = torch.clamp(actions[:, group] + score_offset, -1.0, 1.0)
    return coordinated


def apply_branch_long_finger_velocity_coordination(
    actions: torch.Tensor,
    previous_scores: torch.Tensor,
    max_velocity_deviation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Limit per-step flexion-speed disagreement across the four long fingers.

    Actions remain position targets, so the score difference from the preceding
    control step is the commanded normalized closing velocity.  The thumb is
    intentionally excluded because opposition requires independent motion.
    """
    if actions.shape[-1] != 19:
        raise ValueError("Branch velocity coordination expects the 19-DoF Evolution hand action layout.")
    if previous_scores.shape != (actions.shape[0], 4):
        raise ValueError("Previous long-finger scores must have shape (num_envs, 4).")

    coordinated = actions.clone()
    scores = torch.stack([actions[:, group].mean(dim=-1) for group in _LONG_FINGER_GROUPS], dim=-1)
    raw_velocity = scores - previous_scores
    shared_velocity = raw_velocity.mean(dim=-1, keepdim=True)
    bounded_velocity = torch.clamp(
        raw_velocity,
        shared_velocity - max_velocity_deviation,
        shared_velocity + max_velocity_deviation,
    )
    bounded_scores = previous_scores + bounded_velocity
    for finger_index, group in enumerate(_LONG_FINGER_GROUPS):
        score_offset = (bounded_scores[:, finger_index] - scores[:, finger_index]).unsqueeze(-1)
        coordinated[:, group] = torch.clamp(actions[:, group] + score_offset, -1.0, 1.0)

    next_scores = torch.stack([coordinated[:, group].mean(dim=-1) for group in _LONG_FINGER_GROUPS], dim=-1)
    velocity_spread = (bounded_velocity.max(dim=-1).values - bounded_velocity.min(dim=-1).values)
    return coordinated, next_scores, velocity_spread


def finger_flexion_scores(actions: torch.Tensor) -> torch.Tensor:
    """Return the mean normalized flexion action for thumb through little finger."""
    if actions.shape[-1] != 19:
        raise ValueError("Finger flexion scores expect the 19-DoF Evolution hand action layout.")
    return torch.stack([actions[:, group].mean(dim=-1) for group in _FINGER_GROUPS], dim=-1)
