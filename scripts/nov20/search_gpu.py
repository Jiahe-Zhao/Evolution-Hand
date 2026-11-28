
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,os,json,glob
import numpy as np,mujoco as mj
from mujoco import rollout
from env import episode,OUT
task=sys.argv[1];g=episode(task,gpu=True);meta=next(g);v=g.gi_frame.f_locals;m=v['mm'];m.opt.timestep=1/240
d=mj.MjData(m);mj.mj_resetData(m,d);mj.mj_forward(m,d)
spec=mj.mjtState.mjSTATE_FULLPHYSICS;s0=np.zeros(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,s0,spec)
N=440;B=int(os.getenv('BATCH','48'));ts=np.arange(N)/240
r=lambda a:np.clip(a,0,1)
cur=np.random.default_rng(int(os.getenv("SEED","2027")));files=sorted(glob.glob(str(OUT/f'*{task}_*.npz')));files=[f for f in files if '/opposed_' in f or '/ik_' in f];arr=np.array([np.load(f)['arr'] for f in files]);print(files,flush=True)
def actions(p):
 # approach, close, lift. Single target remains a free body throughout.
 mode,dx,dy,dz,gap,dx2,dy2,z2,lag,add,tipadj,rate=p.T
 t=ts[None,:];u=r((t-.74)/(.27+rate[:,None]));w=r((t-1.17-lag[:,None])/.33)
 u=u*(2-u);w=w*w*(3-2*w);gi=np.clip((22-gap[:,None])*u,0,19.99);k=gi.astype(int)
 mode=mode.astype(int)[:,None];qs=arr[mode,k]*(1-gi+k)[...,None]+arr[mode,k+1]*(gi-k)[...,None]
 qs[:,:,0]+=(dx[:,None]+dx2[:,None]*u)/1000;qs[:,:,2]-=(dy[:,None]+dy2[:,None]*u)/1000;qs[:,:,1]+=(dz[:,None]+z2[:,None]*u)/1000+.055*(1-r((t-.45)/.32))+.036*w
 qs[:,:,10]+=add[:,None]*u;qs[:,:,14]+=tipadj[:,None]*u
 qs*=r(t/.43)[...,None];qs[:,:,1]+=.05*np.sin(np.pi*r(t/.43))
 A=np.zeros((len(p),N,27));A[:,:,:15]=qs;return np.clip(A,*m.jnt_range[:27].T[:,None,None,:])
low=np.array([0,-9,-7,-6,1,-7,-6,-4,-.10,-.45,-.45,-.06]);hi=np.array([len(arr)-.001,9,7,5,13,7,6,4,.10,.45,.45,.10])
inds=[m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_type[i]==0];j=inds[0];start=d.qpos[j:j+3].copy();root=OUT/(task+'_gpu_search');root.mkdir(exist_ok=True);best=-1000;elite=None;archive=[];seedfiles=sorted(root.glob('candidate*.npz'));prev=max([int(f.stem.split('_')[1]) for f in seedfiles],default=-1)+1
if seedfiles:elite=np.stack([np.load(f)['params'] for f in seedfiles[-4:]])
import warp as wp,mujoco_warp as mw
dev='cuda:0'
wm=mw.put_model(m)
wd=mw.put_data(m,d,nworld=B,nconmax=4096,njmax=2048)
# Initialize independent copies from identical reset each round.
initwd=mw.put_data(m,d,nworld=B,nconmax=4096,njmax=2048)
with wp.ScopedCapture(device=dev) as capture:mw.step(wm,wd)
graph=capture.graph
if True:
 for round in range(int(os.getenv('ROUNDS','35'))):
    p=cur.uniform(low,hi,(B,len(low)))
    if elite is not None:
     s=p[B//4:];s[:]=elite[cur.integers(len(elite),size=len(s))]+cur.normal(size=s.shape)*(hi-low)*cur.choice([.015,.035,.07,.15],(len(s),1));s[:]=np.clip(s,low,hi)
    A=actions(p);states=[]
    for name in ('qpos','qvel','qacc_warmstart','time','ctrl'):
     wp.copy(getattr(wd,name),getattr(initwd,name))
    for step in range(N):
     wd.ctrl.assign(np.ascontiguousarray(A[:,step,:],dtype=np.float32));wp.capture_launch(graph)
     if step%8==7:states.append(wd.qpos.numpy().copy())
    S=np.stack(states,axis=1);obj=S[:,:,j:j+3];lift=(obj[:,:,2]-start[2])*1000
    cost=np.minimum(lift[:,-9:],36).mean(1)+.28*np.minimum(lift[:,36:].max(1),45)-.08*np.linalg.norm(obj[:,-1,:2]-start[:2],axis=1)*1000
    if task=='pebble':
     for n in inds[1:]:
      drift=np.linalg.norm(S[:,:,n:3+n]-d.qpos[n:n+3],axis=2).max(1)*1000;cost-=2*np.maximum(drift-4.6,0)
    keep=np.argsort(cost)[-6:];archive.extend((float(cost[i]),p[i].copy()) for i in keep);archive.sort(key=lambda z:-z[0]);seen=set();archive=[v for v in archive if not(tuple(v[1].round(2)) in seen or seen.add(tuple(v[1].round(2))))][:18];elite=np.stack([v[1] for v in archive]);k=keep[-1]
    if cost[k]>best:
      best=cost[k];np.savez(root/f'candidate_{round+prev:03d}.npz',actions=A[k],params=p[k],predicted=obj[k],score=best);print(round,best,lift[k,-1],flush=True)
      if best>45:break
