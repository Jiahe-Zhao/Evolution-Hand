import unittest
import torch
from derl_ppo import Moments,Policy,gae,clipped_losses
class Tests(unittest.TestCase):
 def test_normalization_matches_pooled_statistics(self):
  torch.manual_seed(1);x=torch.randn(300,7,dtype=torch.float64);m=Moments((7,),'cpu');m.update(x[:80]);m.update(x[80:])
  # Include the reference algorithm's epsilon-weighted prior sample.
  expected=x.mean(0)*300/300.0001
  torch.testing.assert_close(m.mean,expected)
  torch.testing.assert_close(m.var,x.var(0,unbiased=False),atol=1e-6,rtol=1e-6)
 def test_terminal_return_does_not_bootstrap(self):
  rewards=torch.tensor([[1.],[2.]]);values=torch.zeros_like(rewards);done=torch.tensor([[False],[True]])
  ret,_=gae(rewards,values,done,torch.tensor([999.]),gamma=.99,lam=1.)
  torch.testing.assert_close(ret,torch.tensor([[2.98],[2.]]))
 def test_rollout_boundary_bootstraps(self):
  ret,_=gae(torch.tensor([[1.]]),torch.tensor([[0.]]),torch.tensor([[False]]),torch.tensor([2.]),gamma=.99,lam=.95)
  torch.testing.assert_close(ret,torch.tensor([[2.98]]))
 def test_random_policy_and_sigma_update(self):
  torch.manual_seed(2027);m=Policy(148);o=torch.randn(64,148);d,v=m(o)
  self.assertEqual(m.actor[0].out_features,64);torch.testing.assert_close(d.stddev,torch.ones_like(d.mean))
  act=d.sample();batch=dict(logp=d.log_prob(act).sum(-1).detach(),adv=torch.randn(64),value=v.detach(),ret=torch.randn(64))
  loss,_,_=clipped_losses(v,d.log_prob(act).sum(-1),d.entropy(),batch)
  opt=torch.optim.Adam(m.parameters(),lr=3e-4,eps=1e-5);opt.zero_grad();loss.backward();opt.step()
  self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.log_std.abs().max()),0)
if __name__=='__main__':unittest.main()
