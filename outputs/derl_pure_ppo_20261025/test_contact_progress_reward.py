import unittest
from types import SimpleNamespace as S
import torch
from contact_progress_reward import progress_gain,clearance_progress,ProgressAdapter
from native_goal_ppo import gae

class RewardInvariants(unittest.TestCase):
 def test_contact_oscillation_cannot_farm(self):
  best=torch.zeros(1,2);total=torch.zeros_like(best)
  for values in [[.2,.4],[.1,.2],[.2,.4],[.8,.9],[0.,0.],[.8,.9]]:
   gain,best=progress_gain(torch.tensor([values]),best);total+=gain
  self.assertTrue(torch.allclose(total,torch.tensor([[.8,.9]])))
 def test_partial_budget_is_bounded(self):
  best=torch.zeros(4,5);total=torch.zeros_like(best)
  for _ in range(100):
   gain,best=progress_gain(torch.rand(4,5)*2,best);total+=gain
  self.assertLessEqual(float(total.max()),1.000001)
 def test_pre_exposure_clearance_has_signal(self):
  # Food remains completely covered at both positions by the larger leaf.
  x=torch.tensor([[.005,0.],[.010,0.]])
  p=clearance_progress(x,torch.tensor([.08,.06]),torch.tensor([.025,.0225]))
  self.assertGreater(float(p[1]),float(p[0]));self.assertGreater(float(p[0]),0.)
 def test_timeout_retains_partial_credit_and_no_repeat(self):
  a=object.__new__(ProgressAdapter);a.device='cpu';a.task='branch';a.progress_weights=[.5,.5,2,3,4];a.progress_names=list('abcde')
  a.best_progress=torch.zeros(1,5);a.current_progress=torch.tensor([[.5,.5,.4,.3,0.]])
  a.event=torch.tensor([False]);a.raw=S(step_dt=1/60,whole_hand_tactile=S(force=torch.zeros(1,5,1,3)))
  value=a.task_reward(torch.zeros(1),torch.zeros(1),torch.tensor([True]),torch.zeros(1))
  self.assertGreater(float(value),2.19)
  repeated=a.task_reward(torch.zeros(1),torch.zeros(1),torch.tensor([False]),torch.zeros(1))
  self.assertLessEqual(float(repeated),0.)
 def test_success_bonus_remains_distinct(self):
  a=object.__new__(ProgressAdapter);a.device='cpu';a.task='branch';a.progress_weights=[1.];a.progress_names=['contact']
  a.best_progress=torch.ones(1,1);a.current_progress=torch.ones(1,1);a.raw=S(step_dt=1/60,whole_hand_tactile=S(force=torch.zeros(1,5,1,3)))
  a.event=torch.tensor([False]);failure=a.task_reward(None,None,None,None)
  a.event=torch.tensor([True]);success=a.task_reward(None,None,None,None)
  self.assertAlmostEqual(float(success-failure),10.,places=5)
 def test_discount_is_applied_in_gae(self):
  rew=torch.tensor([[0.],[1.]]);val=torch.zeros_like(rew);done=torch.tensor([[False],[True]])
  ret,_=gae(rew,val,done,torch.zeros(1),gamma=.999,lam=1.)
  self.assertAlmostEqual(float(ret[0]),.999,places=6)
if __name__=='__main__':unittest.main()
