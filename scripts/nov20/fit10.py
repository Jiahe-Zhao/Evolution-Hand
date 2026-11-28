
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
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);seq=actions[:155].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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
 return (1000*np.linalg.norm(a[...,:3]-target,axis=2)+w*tilt).mean(1)
first=actions[155:405].copy()
beststep(first,lambda S:cost(S,np.r_[up[:2],.066],.07),np.r_[.003,.003,.003,np.zeros(3),np.full(9,.14)],10,'pick',85)
for stage,z in enumerate([.069,.059,.047,.041,.035,.030,.027]):
 target=np.r_[xy,z];N=80;g=np.clip(np.arange(N)/55,0,1);g=g*g*(3-2*g)
 off=target-st[qadr:qadr+3];off=off[[0,2,1]]*[1,1,-1];seedA=np.tile(cmd,(N,1));seedA[:,:3]+=g[:,None]*off
 beststep(seedA,lambda S:cost(S,target,.52),np.r_[.009,.009,.009,.36,.36,.36,np.full(9,.1)],9,'stage'+str(stage),55)
# Lower and release; settle without lateral assistance.
N=165;g=np.clip(np.arange(N)/40,0,1);off=np.r_[xy,.027]-st[qadr:qadr+3];seedA=np.tile(cmd,(N,1));seedA[:,:3]+=g[:,None]*(off[[0,2,1]]*[1,1,-1]);op=np.clip((np.arange(N)-35)/45,0,1);seedA[:,6:15]+=op[:,None]*(actions[150,6:15]-cmd[6:15]);seedA[:,1]+=.055*np.clip((np.arange(N)-80)/70,0,1)
target=np.r_[xy,.027];beststep(seedA,lambda S:cost(S,target,.45),np.r_[.008,.007,.008,.15,.15,.15,np.full(9,.13)],14,'release',35)
np.save(out/'actions.npy',np.concatenate([seq,np.tile(cmd,(330,1))]))
