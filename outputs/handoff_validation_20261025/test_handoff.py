import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from inheritance import load_actor
from handoff import install_guard

class Model(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.a2c_network=torch.nn.Module()
  self.a2c_network.actor_mlp=torch.nn.Sequential(torch.nn.Linear(n,8),torch.nn.ELU())
  self.a2c_network.mu=torch.nn.Linear(8,24)
  self.a2c_network.critic_mlp=torch.nn.Linear(n,8)
  self.a2c_network.value=torch.nn.Linear(8,1)
 def forward(self,feed):
  return {'mus':self.a2c_network.mu(self.a2c_network.actor_mlp(feed['obs']))}

class TestHandoff(unittest.TestCase):
 def test_observation_expansion_preserves_actor_and_new_features_can_learn(self):
  old,new=Model(67),Model(108)
  agent=SimpleNamespace(model=new,optimizer=torch.optim.Adam(new.parameters()),ppo_device='cpu')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'seed.pth';torch.save({'model':old.state_dict()},p)
   result=load_actor(agent,p,{'observation':'wrap67_static_scaling_v1'},{'observation':'wrap108_hold_signed_force_velocity_support_v2'})
   x=torch.randn(32,108)
   torch.testing.assert_close(old({'obs':x[:,:67]})['mus'],new({'obs':x})['mus'],atol=1e-6,rtol=1e-6)
   self.assertTrue(result['observation_expanded_with_zero_weights'])
   new({'obs':x})['mus'].sum().backward()
   self.assertGreater(float(new.a2c_network.actor_mlp[0].weight.grad[:,67:].abs().sum()),0)
 def test_guard_rejects_large_actor_change_and_accepts_small_change(self):
  model=Model(108);a=SimpleNamespace(model=model,optimizer=torch.optim.Adam(model.parameters()),last_lr=1e-7)
  increment=[1.]
  def update(batch):
   with torch.no_grad():model.a2c_network.mu.bias.add_(increment[0])
  a.calc_gradients=update;x=torch.randn(4,108)
  batch={'obs':x,'mu':model({'obs':x})['mus'].detach().clone(),'sigma':torch.ones(4,24)*.01}
  before=model.a2c_network.mu.bias.detach().clone()
  with tempfile.TemporaryDirectory() as d:
   install_guard(a,Path(d));a.calc_gradients(batch)
   torch.testing.assert_close(model.a2c_network.mu.bias,before,atol=0,rtol=0)
   increment[0]=1e-5;a.calc_gradients(batch)
   r=json.loads((Path(d)/'policy_guard.json').read_text())
   self.assertEqual((r['rejected'],r['accepted']),(1,1))
   self.assertGreater(float((model.a2c_network.mu.bias-before).abs().max()),0)
if __name__=='__main__':unittest.main()
