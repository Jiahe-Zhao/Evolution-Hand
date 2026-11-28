
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from place11 import *
from mujoco import rollout
import os
p=out/'audit11.xml';b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k:
  j=b.joint_qd_start[i];b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
m=b.finalize(device='cpu');v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40);mm=v.mj_model;mm.opt.timestep=1/240
d=mj.MjData(mm);mj.mj_forward(mm,d);spec=mj.mjtState.mjSTATE_FULLPHYSICS;s0=np.zeros(mj.mj_stateSize(mm,spec));mj.mj_getState(mm,d,s0,spec)
rr=rollout.Rollout(nthread=12);ds=[mj.MjData(mm) for _ in range(12)]
actions=np.load(out/'audit_actions.npy');qadr=1+mm.jnt_qposadr[-1];goal=R.from_euler('z',yaw).as_quat()[[3,0,1,2]]
def run(st,a):return rr.rollout(mm,ds,np.repeat(st[None],len(a),axis=0),np.ascontiguousarray(a))[0]
def err(s,target):
 o=s[...,qadr:qadr+7];pos=np.linalg.norm(o[...,:3]-target,axis=-1)*1000;ang=np.arccos(np.clip(abs(o[...,3:]@goal),0,1))*2;return pos+50*ang
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);seq=np.load(out/'pick11.npy')[:455].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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
for step,z in enumerate([.067,.067,.064,.056,.046,.039,.034,.030,.027]):
 target=np.r_[xy,z];
 if step<3:target[:2]=np.array([-.093,.005])+(xy-[-.093,.005])*(step+1)/3
 N=75;u=np.clip(np.arange(N)/50,0,1);u=u*u*(3-2*u);B=40
 span=np.array([.013,.014,.013,.35,.35,.35,.12,.12,.12,.12,.12,.12,.12,.12,.12]);mean=np.zeros(len(span));mean[:3]=np.clip((target-st[qadr:qadr+3])[[0,2,1]]*[1,1,-1],-.025,.025);sig=span.copy();best=1e6
 for rnd in range(14):
  delta=rng.normal(mean,sig,(B,len(span)));delta[0]=mean
  A=np.tile(cmd,(B,N,1));A[:,:,:len(span)]+=delta[:,None,:]*u[None,:,None];S=run(st,A)
  score=err(S[:,-16:],target).mean(1)+.3*err(S[:,-1],target)
  k=score.argmin()
  if score[k]<best:best=score[k];a=A[k].copy();end=S[k,-1].copy()
  e=delta[np.argsort(score)[:5]];mean=e.mean(0);sig=np.maximum(.14*span,.7*e.std(0)+.15*sig)
 seq=np.concatenate([seq,a]);cmd=a[-1];st=advance(a);print('diff',np.linalg.norm(st[qadr:qadr+7]-end[qadr:qadr+7]),flush=True);np.save(out/'guided_prefix.npy',seq);print(step,best,st[qadr:qadr+3],flush=True)
# optimize opening fingers and withdrawing wrist; final state must be seated
N=210;u=np.clip(np.arange(N)/45,0,1);ret=np.clip((np.arange(N)-50)/45,0,1)
mean=np.zeros(12);sig=np.r_[np.full(3,.003),np.full(9,.24)];best=1e6
for rd in range(5):
 B=32;P=rng.normal(mean,sig,(B,12));A=np.tile(cmd,(B,N,1));A[:,:,:3]+=P[:,None,:3]*u[None,:,None];A[:,:,1]+=.055*ret;A[:,:,6:15]+=((actions[165,6:15]-cmd[6:15])[None,:]+P[:,3:])[:,None,:]*u[None,:,None];S=run(st,A);score=err(S[:,-120:],np.r_[xy,.027]).mean(1);k=score.argmin()
 if score[k]<best:best=score[k];np.save(out/'guided_full.npy',np.concatenate([seq,A[k]]));print('release',rd,best,flush=True)
 if best<.5:break
 e=P[np.argsort(score)[:6]];mean=e.mean(0);sig=np.maximum(sig*.65,e.std(0))
