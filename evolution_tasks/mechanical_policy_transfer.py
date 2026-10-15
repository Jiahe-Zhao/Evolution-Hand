"""Explicit semantic thumb transfer; does not preserve Cartesian trajectories.

Only enable after binding source/target morphology and checkpoint contracts.
The old zero-to-negative excursion maps to the new zero-to-positive excursion.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThumbTransfer:
    source_lower: float
    target_upper: float
    observation_joint_index: int | None
    observation_joint_count: int
    action_dimension: int
    action_index: int | None

    def __post_init__(self):
        if not (math.isfinite(self.source_lower) and self.source_lower < 0
                and math.isfinite(self.target_upper) and self.target_upper > 0):
            raise ValueError('Requires source [negative, 0] and target [0, positive]')
        if self.observation_joint_index is not None and not 0 <= self.observation_joint_index < self.observation_joint_count:
            raise ValueError('Thumb observation index outside joint layout')
        if self.action_index is not None and not 0 <= self.action_index < self.action_dimension:
            raise ValueError('Thumb action index outside action layout')

    def source_angle_to_target(self, angle):
        return angle * self.target_upper / self.source_lower

    def observation_for_source_policy(self, observation):
        """Torch tensor in target convention -> legacy policy convention.

        Positions are normalized to [-1,1]; velocities are physical angular
        velocities with the same observation scale in source and target.
        The final action_dimension entries must be previous actions.
        """
        if observation.shape[-1] < 2 * self.observation_joint_count + self.action_dimension:
            raise ValueError('Observation is too short for the declared contract')
        result = observation.clone()
        index = self.observation_joint_index
        if index is not None:
            result[..., index] *= -1
            result[..., self.observation_joint_count + index] *= self.source_lower / self.target_upper
        if self.action_index is not None:
            result[..., -self.action_dimension + self.action_index] *= -1
        return result

    def action_for_target_hand(self, action):
        if action.shape[-1] != self.action_dimension:
            raise ValueError('Action does not match the declared controller')
        result = action.clone()
        # Cartesian fingertip targets use action_index=None and stay unchanged.
        if self.action_index is not None:
            result[..., self.action_index] *= -1
        return result
