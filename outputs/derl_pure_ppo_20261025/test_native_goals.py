import tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace as S
import torch
from native_goal_adapter import GoalAdapter
from native_goal_ppo import evaluate,Moments

class FakeAdapter(GoalAdapter):
 def features(self):
  self.held=torch.zeros(self.num_envs,dtype=torch.bool)
  return torch.zeros(self.num_envs,2),torch.full((self.num_envs,),2.)

class NativeGoalsTest(unittest.TestCase):
 def adapter(self,reward):
  raw=S(num_envs=1,device='cpu',cfg=S(action_space=1),max_episode_length=10,
    reset_terminated=torch.tensor([False]),reset_time_outs=torch.tensor([False]),reset_buf=torch.tensor([False]),
    graph_control=S(indices=torch.tensor([0]),lower=torch.tensor([-1.]),upper=torch.tensor([1.])),
    hand=S(data=S(joint_pos=torch.zeros(1,1))),tool_finger_contact_forces=torch.zeros(1,5),
    cone=S(data=S(root_lin_vel_w=torch.zeros(1,3))))
  raw._get_observations=lambda:{'policy':torch.zeros(1,3)}
  raw._get_rewards=lambda:torch.tensor([reward])
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  a=FakeAdapter(S(unwrapped=raw),'strike',Path(self.tmp.name));a.prev_potential=torch.tensor([2.])
  return a,raw
 def test_branch_has_signal_before_contact(self):
  a=object.__new__(GoalAdapter);a.task='branch';a.num_envs=1;a.device='cpu'
  r=S(branch=S(data=S(root_pos_w=torch.tensor([[0.,0.,.1]]))),scene=S(env_origins=torch.zeros(1,3)),
   hand=S(data=S(root_pos_w=torch.zeros(1,3))),branch_contact_sensor=S(data=S(force_matrix_w=torch.zeros(1,1,5,3))),
   cfg=S(branch_contact_force_threshold=.1,branch_success_hold_steps=15,min_long_finger_contacts=2),
   branch_success_streak=torch.zeros(1),step_dt=1/60,cartesian_ik=S(fingertip_positions_world=lambda:torch.zeros(1,5,3)))
  a.raw=r;obs,far=a.features();r.branch.data.root_pos_w[:,2]=.05;obs2,near=a.features()
  self.assertGreater(float(far),0.);self.assertGreater(float(near),float(far));self.assertFalse(torch.equal(obs,obs2))
 def test_strike_grasp_is_not_success(self):
  a,r=self.adapter(250.);reward=r._get_rewards()
  self.assertFalse(a.event.item());self.assertAlmostEqual(reward.item(),-.02,places=5)
  self.assertFalse(r.reset_buf.item())
 def test_success_terminal_potential_zero(self):
  a,r=self.adapter(1000.);reward=r._get_rewards()
  self.assertTrue(a.event.item());self.assertTrue(r.reset_buf.item())
  self.assertEqual(reward.item(),8.)
 def test_timeout_terminal_potential_zero(self):
  a,r=self.adapter(0.);r.reset_time_outs[:]=True
  self.assertEqual(r._get_rewards().item(),-2.)
 def test_configured_discount_matches_potential_shaping(self):
  with patch.dict('os.environ',{'CONTACT_PPO_GAMMA':'.999'}):a,r=self.adapter(0.)
  self.assertAlmostEqual(float(r._get_rewards()),-.002,places=5)
 def test_evaluation_ignores_later_episode(self):
  class Env:
   num_envs=2
   task='branch'
   def reset(self):self.t=0;return torch.zeros(2,1)
   def step(self,action):
    self.t+=1
    return torch.zeros(2,1),torch.ones(2),torch.tensor([True,self.t>=2]),{'episode_success':torch.tensor([self.t>=2,self.t>=2])}
   def contract(self):return {}
  class Net(torch.nn.Module):
   def forward(self,x):return torch.distributions.Normal(torch.zeros(2,1),torch.ones(2,1)),torch.zeros(2)
  with tempfile.TemporaryDirectory() as tmp:
   r=evaluate(Env(),Net(),Moments((1,),'cpu'),S(eval_seed=7,episode_steps=10),Path(tmp),'test')
  self.assertEqual(r['passed'],[False,True]);self.assertEqual(r['first_episode_lengths'],[1,2])
if __name__=='__main__':unittest.main()
