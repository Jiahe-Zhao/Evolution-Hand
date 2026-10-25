import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from features import inverse_rotate,relative_features,object_contact_forces
from adaptive_policy import gaussian_kl,install
from inheritance import load_actor,restore_exploration_for_evaluation

class Model(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.a2c_network=torch.nn.Module();a=self.a2c_network
  a.actor_mlp=torch.nn.Sequential(torch.nn.Linear(n,8),torch.nn.ELU());a.mu=torch.nn.Linear(8,24)
  a.critic_mlp=torch.nn.Linear(n,8);a.value=torch.nn.Linear(8,1);a.sigma=torch.nn.Parameter(torch.zeros(24)-5)
 def forward(self,f):
  a=self.a2c_network;mu=a.mu(a.actor_mlp(f['obs']));return {'mus':mu,'sigmas':a.sigma.exp().expand_as(mu)}
class Tests(unittest.TestCase):
 def test_exported_exploration_restored(self):
  model=Model(133);old=model.a2c_network.sigma.detach().clone();agent=SimpleNamespace(model=model,ppo_device='cpu')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'saved.pth';torch.save({'model':model.state_dict()},path)
   with torch.no_grad():model.a2c_network.sigma.add_(.2)
   restore_exploration_for_evaluation(agent,path)
   torch.testing.assert_close(model.a2c_network.sigma,old,atol=0,rtol=0)
 def test_frames(self):
  q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
  force=torch.tensor([[[0.,2.,0.],[0.,-3.,0.]]])
  local=object_contact_forces(q,force)
  torch.testing.assert_close(local[0,:,0],torch.tensor([2.,-3.]))
  parent=torch.tensor([[.7,0.,0.]])
  child=torch.zeros(1,13);child[:,:3]=parent+torch.tensor([[0.,.05,0.]]);child[:,3:7]=q
  f=relative_features(parent,q,torch.zeros(1,3),torch.zeros(1,3),child,torch.zeros(1,5,3))
  torch.testing.assert_close(f[0,:3],torch.tensor([1.,0.,0.]),atol=1e-6,rtol=1e-6)
  torch.testing.assert_close(f[0,6:10],torch.tensor([1.,0.,0.,0.]),atol=1e-6,rtol=1e-6)
  self.assertEqual(f.shape,(1,25))
 def test_gaussian_kl_detects_variance_only_change(self):
  mu=torch.zeros(5,24);a=torch.full_like(mu,.0001);b=a*1.2
  reference=torch.distributions.kl_divergence(torch.distributions.Normal(mu,a),torch.distributions.Normal(mu,b)).sum(-1).mean()
  torch.testing.assert_close(gaussian_kl(mu,a,mu,b).float(),reference,atol=1e-5,rtol=1e-5)
  self.assertGreater(float(reference),.01)
 def test_migration(self):
  for n,version in [(67,'wrap67_static_scaling_v1'),(108,'wrap108_hold_signed_force_velocity_support_v2')]:
   old,new=Model(n),Model(133);a=SimpleNamespace(model=new,optimizer=torch.optim.Adam(new.parameters()),ppo_device='cpu')
   with tempfile.TemporaryDirectory() as d:
    path=Path(d)/'p.pth';torch.save({'model':old.state_dict()},path)
    load_actor(a,path,{'observation':version},{'observation':'wrap133_local_object_hand_v3'})
   x=torch.randn(10,133);torch.testing.assert_close(old({'obs':x[:,:n]})['mus'],new({'obs':x})['mus'])
 def test_reject_restores_sigma_and_actor(self):
  m=Model(133);a=SimpleNamespace(model=m,optimizer=torch.optim.Adam(m.parameters()),epoch_num=1,last_lr=1e-7)
  def huge(batch):
   with torch.no_grad():m.a2c_network.sigma.add_(2.)
  a.calc_gradients=huge;x=torch.randn(4,133);p=m({'obs':x});batch={'obs':x,'mu':p['mus'].detach(),'sigma':p['sigmas'].detach().clone()}
  old=m.a2c_network.sigma.detach().clone()
  with tempfile.TemporaryDirectory() as d:
   state=install(a,Path(d),warmup=0);a.calc_gradients(batch)
   self.assertEqual(state['rejected'],1);torch.testing.assert_close(m.a2c_network.sigma,old,atol=0,rtol=0)
if __name__=='__main__':unittest.main()
