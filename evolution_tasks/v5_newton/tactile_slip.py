"""Virtual tactile sampling. Friction utilization is a warning, not slip proof."""
import math
import torch

class SlipDetector:
    def __init__(self, history=4, mu=.8, grid_mm=1., dt=1/60, speed_threshold=.005):
        if min(mu, grid_mm, dt, speed_threshold) <= 0:
            raise ValueError('Positive friction, pitch, timestep and speed threshold required')
        self.mu, self.grid_mm, self.dt = mu, grid_mm, dt
        self.speed_threshold = speed_threshold
        self.previous = None
        self.valid = None

    def reset(self, env_ids=None):
        if env_ids is None:
            self.previous = self.valid = None
        elif self.valid is not None:
            self.valid[env_ids] = False

    def update(self, normal, friction, contact_xy, relative_velocity=None):
        """Forces N; coordinates m; tangential relative velocity m/s.
        Centroid drift alone may indicate rolling, not sliding.
        Call once per measured timestep, never from observation reads.
        """
        if contact_xy.shape != normal.shape + (2,) or friction.shape[:-1] != normal.shape:
            raise ValueError('Incompatible tactile tensor dimensions')
        contact = normal > .02
        ratio = torch.linalg.vector_norm(friction, dim=-1) / normal.clamp_min(1e-8)
        if self.previous is None:
            drift = torch.zeros_like(normal)
        else:
            drift = torch.linalg.vector_norm(contact_xy-self.previous, dim=-1) / self.dt
            drift = torch.where(self.valid & contact, drift, 0.)
        speed = torch.zeros_like(normal) if relative_velocity is None else torch.linalg.vector_norm(relative_velocity, dim=-1)
        self.previous = contact_xy.detach().clone()
        self.valid = contact.detach().clone()
        self.last = dict(slip=contact & (speed > self.speed_threshold),
                         friction_warning=contact & (ratio > .9*self.mu),
                         centroid_motion=contact & (drift > self.speed_threshold),
                         tangential_speed_m_s=speed, friction_utilization=ratio/self.mu)
        return self.last['slip']

    def rasterize(self, normal, contact_xy, size_mm=12, valid=None):
        """Deposit [env,finger,point] loads. Drop outside field; sum shared cells."""
        if normal.ndim == 2:
            normal=normal.unsqueeze(-1); contact_xy=contact_xy.unsqueeze(-2)
            if valid is not None: valid=valid.unsqueeze(-1)
        if normal.ndim != 3 or contact_xy.shape != normal.shape+(2,):
            raise ValueError('Expected [environment,finger,point] loads')
        cells=size_mm/self.grid_mm
        if not math.isclose(cells,round(cells)) or cells<1:
            raise ValueError('Sensor width must contain integer cells')
        n=round(cells); pitch=self.grid_mm*.001
        inside=torch.isfinite(contact_xy).all(-1) & torch.isfinite(normal) & (normal>=0)
        ij=torch.floor(torch.nan_to_num(contact_xy)/pitch+n/2).long()
        inside &= (ij>=0).all(-1) & (ij<n).all(-1)
        if valid is not None: inside &= valid
        index=(ij[...,1]*n+ij[...,0]).clamp(0,n*n-1)
        out=torch.zeros(*normal.shape[:2],n*n,device=normal.device,dtype=normal.dtype)
        out.scatter_add_(-1,index,torch.where(inside,normal,0.))
        return out.reshape(*normal.shape[:2],n,n)
