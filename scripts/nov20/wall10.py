
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
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);src=np.load(out/'shoot0_0_actions.npy');seq=src[:543].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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



N=104;B=64;u=np.clip(np.arange(N)/35,0,1);g=u*u*(3-2*u);sc=np.r_[.0016,.0015,.0016,.042,.042,.042,.04,.04,.04,.04];mu=np.zeros(10);sig=sc.copy();rng=np.random.default_rng(209);best=-1e6
for rd in range(15):
 P=rng.normal(mu,sig,(B,10));P[0]=mu;A=np.tile(src[543:543+N],(B,1,1));A[:,:,:6]+=P[:,None,:6]*g[None,:,None]
 for i,j in enumerate([8,10,12,14]):A[:,:,j]+=P[:,None,6+i]*g
 S=run(st,A);o=S[:,:,qadr:qadr+7];qw,qx,qy,qz=np.moveaxis(o[:,:,3:],-1,0);norm=np.stack([2*(qx*qz+qw*qy),2*(qy*qz-qw*qx),1-2*(qx*qx+qy*qy)],axis=-1)
 diff=o[:,:,:3]-np.r_[xy,zt];tipz=-(diff*norm).sum(2);dep=tipz+H;h=np.minimum(H,tipz+rad*np.sqrt(np.maximum(0,1-norm[:,:,2]**2)));bot=o[:,:,:2]-xy-H*norm[:,:,:2];top=o[:,:,:2]-xy+h[:,:,None]*norm[:,:,:2];nz=norm[:,:,2].clip(.1)
 gap=(hole*np.cos(np.pi/12)-(np.maximum(np.linalg.norm(bot,axis=2),np.linalg.norm(top,axis=2))+rad)/nz)*1000
 worst=np.where(h>=-H,gap,3.).min(1);score=worst-3*np.maximum(0,10.8-dep[:,-16:].min(1)*1000)
 ix=np.argsort(score)[::-1];e=P[ix[:10]];mu=e.mean(0);sig=np.maximum(sc*.13,e.std(0))
 if score[ix[0]]>best:
  best=score[ix[0]];print(rd,best,flush=True)
  for j in range(3):np.save(out/f'wall{rd}_{j}_actions.npy',np.concatenate([seq,A[ix[j]]]))
 if best>.33:break
