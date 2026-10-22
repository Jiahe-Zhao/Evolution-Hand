"""Gaussian KL without an additive epsilon that dominates small variances."""
import torch

def policy_kl(mu0,sigma0,mu1,sigma1,reduce=True):
 s0=sigma0.float().clamp_min(1e-12);s1=sigma1.float().clamp_min(1e-12)
 ratio=s0/s1;delta=(mu1.float()-mu0.float())/s1
 kl=(torch.log(s1)-torch.log(s0)+.5*(ratio.square()+delta.square()-1)).sum(-1)
 return kl.mean() if reduce else kl

def verify():
 rows=[]
 for scale in (1.,.003,.0005,.0001):
  mu=torch.zeros(8,24);std=torch.ones_like(mu)*scale
  identical=policy_kl(mu,std,mu,std)
  shifted=policy_kl(mu,std,mu+.1*std,std)
  reference=torch.distributions.kl_divergence(torch.distributions.Normal(mu,std),torch.distributions.Normal(mu+.1*std,std)).sum(-1).mean()
  assert abs(float(identical))<1e-7
  assert torch.allclose(shifted,reference,atol=2e-6,rtol=2e-5)
  assert float(shifted)>=0
  rows.append(dict(std=scale,identical_kl=float(identical),shifted_kl=float(shifted),reference=float(reference)))
 return rows
if __name__=='__main__':
 import json
 print(json.dumps(verify(),indent=2))
