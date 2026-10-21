"""Exercise temporal failure contracts with transient, recovery and persistent motion."""
import tempfile,types
from pathlib import Path
import torch
from rl_tolerance import install
class MockForage:
 def _reset_idx(self,ids):self.invalid_leaf_motion.zero_()
 def _get_rewards(self):
  speeds=torch.stack([x.data.root_lin_vel_w.norm(dim=-1) for x in (self.leaf_one,self.leaf_two)],-1)
  self.invalid_leaf_motion|=(speeds>2).any(-1)
  return torch.zeros(1)
 def _get_dones(self):return self.invalid_leaf_motion|(self.clear_streak>=10),torch.zeros(1,dtype=torch.bool)
 def _leaf_distances(self):return torch.tensor([[.05,.05]])
cfg=types.SimpleNamespace(sim=types.SimpleNamespace(dt=1/120),decimation=2,maximum_valid_leaf_distance=.5,maximum_valid_leaf_speed=2.,leaf_success_hold_steps=10)
e=MockForage();e.cfg=cfg;e.device='cpu';e.num_envs=1;e.invalid_leaf_motion=torch.zeros(1,dtype=torch.bool);e.clear_streak=torch.zeros(1);e.food_pos=torch.tensor([[0.,.09,.06]]);e.hand=types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.zeros((1,21))))
for n in ['leaf_one','leaf_two']:setattr(e,n,types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_w=torch.zeros((1,3)))))
with tempfile.TemporaryDirectory() as d:
 install(MockForage,cfg,'forage',Path(d));e._reset_idx(None)
 e.leaf_one.data.root_lin_vel_w[0,0]=2.1;e._get_rewards();assert not e._get_dones()[0].item()
 e.leaf_one.data.root_lin_vel_w.zero_();e._get_rewards();assert not e.invalid_leaf_motion.item();assert not e._get_dones()[0].item()
 e.leaf_one.data.root_lin_vel_w[0,0]=2.1
 for _ in range(8):e._get_rewards();assert not e._get_dones()[0].item()
 e._get_rewards();assert e._get_dones()[0].item()
 e._reset_idx(None);e.leaf_one.data.root_lin_vel_w.zero_();e.clear_streak[:]=10;e._get_rewards();assert e._get_dones()[0].item()
 e.clear_streak.zero_();e.leaf_one.data.root_lin_vel_w[0,0]=float('nan');e._get_rewards();assert e._get_dones()[0].item()
print('PASS: transient speed recovers; persistent speed terminates; success preserved; NaN terminates')
