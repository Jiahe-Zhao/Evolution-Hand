"""Equal-budget differentiable kinematics search for both imported USD hands."""
import json,math,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def run_search(directory,out):
 d=Path(directory);out=Path(out);out.mkdir(exist_ok=False)
 k=json.loads((d/'kinematics.json').read_text());inventory=json.loads((d/'asset_inventory.json').read_text());frame=json.loads((d/'anatomical_frame.json').read_text())
 device='cuda';T=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
 names=k['joint_names'];act=inventory['actuated'];indices=[names.index(n) for n in act];baseq=T(k['q']);limits=T(inventory['joint_limits']);lo=limits[indices,0];hi=limits[indices,1]
 root=k['body_names'][0];rootmat=np.eye(4);rootmat[:3,:3]=Rotation.from_quat(np.array(k['body_quat'][0])[[1,2,3,0]]).as_matrix();rootmat[:3,3]=k['body_pos'][0];rootmat=T(rootmat)
 ordered=[];seen={root};todo=list(k['joints'])
 while todo:
  progress=False
  for joint in list(todo):
   if joint['parent'] in seen:
    ordered.append(joint);seen.add(joint['child']);todo.remove(joint);progress=True
  if not progress:raise RuntimeError('Disconnected FK joints '+str([j['name'] for j in todo]))
 prep=[(j,T(j['frame0']),T(np.linalg.inv(j['frame1']))) for j in ordered]
 def forward(q):
  n=len(q);world={root:rootmat[None].expand(n,-1,-1)}
  for j,a,b in prep:
   rot=torch.eye(4,device=device).repeat(n,1,1)
   if j['kind']=='PhysicsRevoluteJoint':
    theta=q[:,names.index(j['name'])];c=theta.cos();s=theta.sin();axis='XYZ'.index(j['axis']);u=(axis+1)%3;v=(axis+2)%3
    rot[:,u,u]=c;rot[:,v,v]=c;rot[:,u,v]=-s;rot[:,v,u]=s
   world[j['child']]=world[j['parent']]@a@rot@b
  return world
 actual=forward(baseq[None]);errors={name:float((actual[name][0,:3,3]-T(pos)).norm()) for name,pos in zip(k['body_names'],k['body_pos'])}
 if max(errors.values())>1e-4:raise RuntimeError('FK does not match imported asset '+str(errors))
 offsets=np.array([[.023,-.005+y,.032+z] for y in np.linspace(-.03,.03,7) for z in np.linspace(-.03,.03,7)])
 rotation=np.asarray(frame['rotation']);origin=np.asarray(frame['origin_world'])+np.asarray(k['env_origin'])
 centers=T(origin+offsets@rotation.T).repeat_interleave(16,0);N=len(centers)
 y=T(rotation[:,1]);z=T(rotation[:,2]);axis=T(rotation[:,0]);radii=T([.009+min(t['radius'],.012) for t in k['tips']])
 torch.manual_seed(2028)
 q=torch.nn.Parameter(lo+(hi-lo)*torch.rand(N,len(indices),device=device));phi=torch.nn.Parameter(torch.arange(N,device=device).float()%16*(2*math.pi/16))
 devi=torch.nn.Parameter(torch.zeros(N,4,device=device));opt=torch.optim.Adam([q,phi,devi],lr=.035)
 def full():
  a=baseq.repeat(N,1);a[:,indices]=q
  if inventory['variant']=='shadow':
   for prefix in ['FF','MF','RF','LF']:
    passive=names.index('robot0_'+prefix+'J0');active=names.index('robot0_'+prefix+'J1')
    a[:,passive]=(a[:,active]*(.00805/.00705)).clamp(limits[passive,0],limits[passive,1])
  return a
 def objective():
  world=forward(full());tips=torch.stack([(world[t['body']][:,:3,:3]@T(t['center'])[None,:,None]).squeeze(-1)+world[t['body']][:,:3,3] for t in k['tips']],1)
  angles=torch.cat((phi[:,None],phi[:,None]+math.pi+devi),1)
  target=centers[:,None]+radii[None,:,None]*(angles.cos()[:,:,None]*y+angles.sin()[:,:,None]*z)
  delta=tips-target;axial=(delta*axis).sum(-1);radial=delta-axial[:,:,None]*axis
  error=radial.norm(dim=-1)+torch.relu(axial.abs()-.08)
  loss=error[:,0].square()+error[:,1:].square().topk(2,largest=False).values.mean(-1)+.02*error[:,1:].square().mean(-1)
  return loss,error,tips
 for i in range(400):
  opt.zero_grad();loss,_,_=objective();loss.mean().backward();opt.step()
  with torch.no_grad():q.clamp_(lo,hi);devi.clamp_(-math.radians(80),math.radians(80))
 loss,error,tips=objective();chosen=[int(loss[i*16:(i+1)*16].argmin())+i*16 for i in range(49)]
 for i in loss.argsort().tolist():
  if i not in chosen:chosen.append(i)
  if len(chosen)==64:break
 quat=Rotation.from_matrix(rotation@Rotation.from_euler('y',90,degrees=True).as_matrix()).as_quat()[[3,0,1,2]].tolist()
 result=dict(variant=inventory['variant'],fk_max_error_m=max(errors.values()),search_initializations=N,optimizer_steps=400,local_offsets=offsets.tolist(),chosen=chosen,joint_names=names,actuated=act,joint_target=full()[chosen].detach().cpu().tolist(),centers_world=(centers[chosen]-T(k['env_origin'])).cpu().tolist(),quat_world=quat,radial_errors_m=error[chosen].detach().cpu().tolist(),loss=loss[chosen].detach().cpu().tolist(),shadow_passive_kinematic_assumption='J0=clip(J1*0.00805/0.00705); zero tendon stretch proxy only; physical replay uses native tendons' if inventory['variant']=='shadow' else None)
 (out/'candidates.json').write_text(json.dumps(result,indent=2));print(inventory['variant'],'FK',result['fk_max_error_m'],'best',min(result['loss']))
if __name__=='__main__':run_search(sys.argv[1],sys.argv[2])
