
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
seed=int(os.getenv('SEED','45'));rng=np.random.default_rng(seed);src=np.load(out/'shoot_final_actions.npy');seq=src[:609].copy();st=run(s0,seq[None])[0,-1];cmd=seq[-1].copy()
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

# Check actual contacting geoms at accepted depth, not center height alone.
names=[mm.geom(i).name for i in range(mm.ngeom)]
handids=set(i for i,n in enumerate(names) if '/link_' in n);itemids=set(i for i,n in enumerate(names) if '/item/' in n)
def touched(d):
 g=d.contact.geom;return any(d.contact.dist[i]<=0 and ((a in handids and b in itemids) or (b in handids and a in itemids)) for i,(a,b) in enumerate(g))
N=52;B=48;tt=np.arange(N);u=np.clip(tt/14,0,1);sett=np.r_[xy,.027];mu=np.zeros(10);sc=np.array([.004,.006,.004,.15,.15,.15,.16,.16,.18,.18]);sig=sc.copy();best=1e6
for rd in range(24):
 P=rng.normal(mu,sig,(B,10));A=np.tile(src[len(seq):len(seq)+N],(B,1,1));A[:,:,:6]+=P[:,None,:6]*u[None,:,None]
 for l,jj in enumerate([8,10,12,14]):A[:,:,jj]+=P[:,6+l,None]*u
 S=run(st,A);rank=[]
 for i in range(B):
  d=mj.MjData(mm);mj.mj_setState(mm,d,st,spec);d.qacc_warmstart[:]=v.mj_data.qacc_warmstart
  val=1e4;matched=False
  for k in range(N):
   d.ctrl[:]=A[i,k].astype(np.float32);mj.mj_step(mm,d)
   if k<7 or not touched(d):continue
   q=d.qpos[-7:];rot=R.from_quat(q[[4,5,6,3]]);angle=np.rad2deg((R.from_euler('z',yaw).inv()*rot).magnitude());pts=rot.apply(np.c_[outline,np.full(5,-.004)])+q[:3];dep=1000*(.029-max(pts[:,2]));off=1000*np.linalg.norm(q[:2]-xy);val=min(val,max(0,4.9-dep)*3+max(0,angle-9)*.25+max(0,off-1.8)*2);matched|=dep>=4.85 and angle<9.8 and off<1.95
  endcost=err(S[i,-1],sett);cost=val+endcost*.3;rank.append(cost)
  if matched and cost<best:
   best=cost;aa=A[i];ret=np.tile(aa[-1],(200,1));ret[:,1]+=.04*np.clip(np.arange(200)/60,0,1);np.save(out/'press_full.npy',np.concatenate([seq,aa,ret]));print('matched',rd,i,best,flush=True)
 if best<1.:break
 elite=P[np.argsort(rank)[:10]];mu=elite.mean(0);sig=np.maximum(sc*.22,elite.std(0));print('rd',rd,min(rank),flush=True)
