
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from stack09_multi import *
from mujoco import rollout
import os
p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k:
  j=b.joint_qd_start[i];b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
m=b.finalize(device='cpu');v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40);mm=v.mj_model;mm.opt.timestep=1/240
d=mj.MjData(mm);mj.mj_forward(mm,d);spec=mj.mjtState.mjSTATE_FULLPHYSICS;s0=np.zeros(mj.mj_stateSize(mm,spec));mj.mj_getState(mm,d,s0,spec)
rr=rollout.Rollout(nthread=12);ds=[mj.MjData(mm) for _ in range(12)]
actions=np.load(OUT/'stack09/custom_actions.npy');qadr=1+mm.jnt_qposadr[-5];lowadr=1+mm.jnt_qposadr[-4]
def run(st,a):return rr.rollout(mm,ds,np.repeat(st[None],len(a),axis=0),np.ascontiguousarray(a))[0]
def err(s,target):
 o=s[...,qadr:qadr+7];pos=np.linalg.norm(o[...,:3]-target,axis=-1)*1000;ang=np.arccos(np.clip(abs(o[...,3:]@goal),0,1))*2;return pos+50*ang
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);seq=actions[:710].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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

def beststep(seedA,evalfn,scale,rounds,name,phase):
 global st,seq,cmd
 N=len(seedA);B=32;g=np.clip(np.arange(N)/phase,0,1);g=g*g*(3-2*g);mean=np.zeros(len(scale));sig=scale.copy();best=1e9
 for rd in range(rounds):
  P=rng.normal(mean,sig,(B,len(mean)));P[0]=mean;P[1]=0
  A=np.tile(seedA,(B,1,1));A[:,:,:len(scale)]+=P[:,None,:]*g[None,:,None]
  S=run(st,A);score=evalfn(S);k=score.argmin()
  if score[k]<best:best=score[k];pick=A[k].copy()
  elite=P[np.argsort(score)[:6]];mean=elite.mean(0);sig=np.maximum(scale*.15,elite.std(0))
  if best<1:break
 seq=np.concatenate([seq,pick]);st=advance(pick);cmd=pick[-1];np.save(out/(name+'.npy'),seq);print(name,best,st[qadr:qadr+7],st[lowadr:lowadr+3],flush=True)
def cost(S,target,w=.16):
 a=S[:,-18:,qadr:qadr+7];tilt=np.rad2deg(np.arccos(np.clip(1-2*(a[...,4]**2+a[...,5]**2),-1,1)));bottom=S[:,:,lowadr:lowadr+3]
 return (1000*np.linalg.norm(a[...,:3]-target,axis=2)+w*tilt).mean(1)+3*np.maximum(0,np.linalg.norm(bottom-bp,axis=2).max(1)*1000-7)

g=np.clip(np.arange(200)/32,0,1);sig=np.r_[.005,.006,.005,.09,.09,.09,np.full(9,.12)];mu=np.zeros(len(sig));sc=sig.copy();best=1e9
for rd in range(12):
 B=32;P=rng.normal(mu,sig,(B,15));P[0]=mu;A=np.tile(actions[710:910],(B,1,1));A[:,:,:15]+=P[:,None,:]*g[None,:,None];S=run(st,A);vls=cost(S,np.r_[xy,.027],.4);bottom=S[:,-16:,lowadr:lowadr+7];vls+=np.linalg.norm(bottom[...,:3]-bp,axis=2).mean(1)*500
 ix=np.argsort(vls)
 for j in range(3):
  k=ix[j];q=np.concatenate([seq,A[k],np.tile(A[k,-1],(425,1))]);np.save(out/f'candidate{rd}_{j}_actions.npy',q)
 if vls[ix[0]]<best:best=vls[ix[0]];np.save(out/'adjusted_actions.npy',np.load(out/f'candidate{rd}_0_actions.npy'));print(rd,best,flush=True)
 if best<1.7:break
 elite=P[ix[:6]];mu=elite.mean(0);sig=np.maximum(sc*.15,elite.std(0))
