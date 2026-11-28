
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from bead10 import *
from mujoco import rollout
import os
p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k:
  j=b.joint_qd_start[i];b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
m=b.finalize(device='cpu');v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;mm.opt.timestep=1/240
d=mj.MjData(mm);mj.mj_forward(mm,d);spec=mj.mjtState.mjSTATE_FULLPHYSICS;s0=np.zeros(mj.mj_stateSize(mm,spec));mj.mj_getState(mm,d,s0,spec)
rr=rollout.Rollout(nthread=12);ds=[mj.MjData(mm) for _ in range(12)]
actions=np.load(OUT/'stack09/custom_actions.npy');qadr=1+mm.jnt_qposadr[-1];lowadr=1+mm.jnt_qposadr[-1]
def run(st,a):return rr.rollout(mm,ds,np.repeat(st[None],len(a),axis=0),np.ascontiguousarray(a))[0]
def err(s,target):
 o=s[...,qadr:qadr+7];pos=np.linalg.norm(o[...,:3]-target,axis=-1)*1000;ang=np.arccos(np.clip(abs(o[...,3:]@goal),0,1))*2;return pos+50*ang
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);seq=np.load(out/'actions.npy')[:525].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
print('START',st[qadr:qadr+3],flush=True)
state=m.state();nxt=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,state);ctrl=m.control();bas=ctrl.joint_target_q.numpy()
names=[k.split('/')[-1] for k in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};ids=[b.joint_qd_start[i] for i,k in enumerate(names) if k in valid and k.startswith(('wrist_','diagnostic_wrist_','link_'))]
def advance(a):
 global state,nxt
 for act in a:
  bas[ids]=act;ctrl.joint_target_q.assign(bas);state.clear_forces();v.step(state,nxt,ctrl,None,1/240);state,nxt=nxt,state
 st=np.zeros_like(s0);mj.mj_getState(mm,v.mj_data,st,spec);return st
st=advance(seq)
def run(st,a):
 return rr.rollout(mm,ds,np.tile(st,(len(a),1)),np.ascontiguousarray(a.astype(np.float32).astype(float)),initial_warmstart=np.tile(v.mj_data.qacc_warmstart,(len(a),1)))[0]


N=130;B=48;t=np.arange(N);rng=np.random.default_rng(87);u=np.clip(t/75,0,1);u=u*u*(3-2*u);op=np.clip((t-77)/35,0,1);mu=np.r_[-.001,-.032,0.,0.,0.,0.,np.zeros(9)];sc=np.r_[.004,.005,.004,.1,.1,.1,np.full(9,.13)];sig=sc.copy();best=1e6
for rd in range(38):
 P=rng.normal(mu,sig,(B,15));P[0]=mu;A=np.tile(cmd,(B,N,1));A[:,:,:15]+=P[:,None,:]*u[None,:,None]
 S=run(st,A);o=S[:,:,qadr:qadr+7];qw,qx,qy,qz=np.moveaxis(o[:,:,3:],-1,0);nz=1-2*(qx*qx+qy*qy);norm=np.stack([2*(qx*qz+qw*qy),2*(qy*qz-qw*qx),nz],axis=-1)
 diff=o[:,:,:3]-np.r_[xy,zt];tipz=-(diff*norm).sum(2);dep=tipz+H
 # Axis offset of rod at bottom of bead, and top of immersed segment.
 bot=o[:,:,:2]-xy-H*norm[:,:,:2];top=o[:,:,:2]-xy+np.minimum(H,tipz)[:,:,None]*norm[:,:,:2];radial=np.maximum(np.linalg.norm(bot,axis=2),np.linalg.norm(top,axis=2))*1000
 over=np.where(dep>-.0002,np.maximum(0,radial-1.),0).max(1)
 cost=over*2.8+np.linalg.norm(o[:,-14:,:3]-np.r_[xy,.032],axis=2).mean(1)*1000+np.rad2deg(np.arccos(nz[:,-14:].clip(-1,1))).mean(1)*.14
 ix=np.argsort(cost)
 if cost[ix[0]]<best:
  best=cost[ix[0]];k=ix[0];print(rd,best,over[k],o[k,-1,:3],flush=True)
  for j in range(3):
   k=ix[j];a=A[k];ret=np.tile(a[-1],(220,1));ret[:,6:15]+=np.clip(np.arange(220)/46,0,1)[:,None]*(actions[145,6:15]-ret[0,6:15]);ret[:,1]+=.05*np.clip((np.arange(220)-55)/60,0,1);np.save(out/f'shoot{rd}_{j}_actions.npy',np.concatenate([seq,a,ret]))
 if best<1.1:break
 elite=P[ix[:8]];mu=elite.mean(0);sig=np.maximum(sc*.13,elite.std(0))
