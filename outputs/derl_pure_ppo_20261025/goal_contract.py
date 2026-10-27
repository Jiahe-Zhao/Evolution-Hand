"""Grasp goals shared by observations, rewards and success evaluation (SI units)."""
import torch

class LiftGoals:
    def __init__(self, count, device, seed, heights=(.025,.040,.055), tolerance=.0075):
        if not heights or any(h <= .015 or h > .1 for h in heights):
            raise ValueError('Lift heights must be in (0.015, 0.1] metres')
        if not 0 < tolerance < min(heights):
            raise ValueError('Invalid height tolerance')
        self.levels=torch.tensor(heights,device=device)
        self.target=torch.empty(count,device=device)
        self.tolerance=tolerance
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.reset(torch.arange(count,device=device),evaluation=True)

    def reset(self, ids, evaluation=False):
        choice=ids % len(self.levels) if evaluation else torch.randint(len(self.levels),(len(ids),),device=ids.device,generator=self.generator)
        self.target[ids]=self.levels[choice]

    def error(self, rise):
        return self.target-rise

    def reached(self, rise):
        return self.error(rise).abs() <= self.tolerance

    def lift_reward(self, rise):
        progress=(rise/self.target).clamp(0,1)
        excess=(self.error(rise).abs()-self.tolerance).clamp_min(0)
        return progress*torch.exp(-excess/self.tolerance)

    def features(self, rise):
        # Positive error asks for upward motion. No world-origin coordinates.
        return torch.stack((self.target/.05,self.error(rise)/.05,
                            torch.full_like(rise,self.tolerance/.05),
                            torch.full_like(rise,.1/.1)),dim=-1)

    def contract(self):
        return dict(version='lift_goal_v1',task='grasp',height_levels_m=self.levels.tolist(),height_tolerance_m=self.tolerance,
                    observation_names=['target_rise_div_0.05','height_error_div_0.05','tolerance_div_0.05','speed_limit_div_0.1'],
                    success='old held predicate AND height within tolerance, continuously 3 seconds at episode end, no evaluation falls',
                    target_frame='world gravity axis relative to initial object height',evaluation_assignment='environment index modulo target count',
                    orientation='existing upright requirement; yaw unconstrained',training_assignment='uniform at each environment reset')
