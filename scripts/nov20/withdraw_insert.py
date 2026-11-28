
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
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);seq=np.load(out/'press_final_actions.npy')[:655].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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

# Search coordinated wrist descent and gradual finger opening. Free object only.
N=110;B=60;t=np.arange(N)[None,:];mu=np.array([.0,.025,.0,0.,0.,0.,1.,1.,0.,0.,0.]);scale=np.array([.009,.012,.009,.35,.35,.35,.50,.5,.28,.28,14]);sig=scale.copy();best=1e6
for rd in range(36):
 P=rng.normal(mu,sig,(B,len(mu)));P[:5]=rng.normal(mu,scale,(5,len(mu)));u=np.clip(t/32,0,1);u=u*u*(3-2*u);op=np.clip((t-P[:,-1,None])/24,0,1);op=op*op*(3-2*op)
 A=np.tile(cmd,(B,N,1));A[:,:,:6]+=P[:,None,:6]*u[...,None]
 A[:,:,6:11]+=(actions[155,6:11]-cmd[6:11])[None,None,:]*P[:,None,6,None]*op[...,None]
 A[:,:,11:15]+=(actions[155,11:15]-cmd[11:15])[None,None,:]*P[:,None,7,None]*op[...,None]
 A[:,:,10]+=P[:,None,8]*op;A[:,:,14]+=P[:,None,9]*op
 S=run(st,A);o=S[:,-12:,qadr:qadr+7];pos=np.linalg.norm(o[:,:,:2]-xy,axis=2)*1000;ang=np.arccos(np.clip(abs(o[:,:,3:]@goal),0,1))*2
 cost=(pos+2*np.abs(o[:,:,2]-.027)*1000+16*ang).mean(1);i=cost.argmin()
 if cost[i]<best:
  best=cost[i];aa=A[i];retract=np.tile(aa[-1],(220,1));retract[:,1]+=.055*np.clip(np.arange(220)/65,0,1);np.save(out/'withdraw_full.npy',np.concatenate([seq,aa,retract]));print(rd,best,o[i,-1,:3],flush=True)
 if best<1.8:break
 e=P[np.argsort(cost)[:9]];mu=e.mean(0);sig=np.maximum(scale*.15,e.std(0))
