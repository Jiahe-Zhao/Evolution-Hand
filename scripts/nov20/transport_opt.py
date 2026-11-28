
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os,sys,json,re
from env import episode,OUT,np,mj
from scipy.spatial.transform import Rotation as R
import warp as wp,mujoco_warp as mw
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
@wp.kernel
def sensor(geom:wp.array[wp.vec2i],dist:wp.array[float],world:wp.array[int],nat:wp.array[int],codes:wp.array[int],frame:wp.array[wp.mat33],friction:wp.array[vec5],dim:wp.array[int],addr:wp.array2d[int],ef:wp.array2d[float],nj:int,cone:int,reduced:wp.array2d[float]):
 i=wp.tid()
 if i>=nat[0] or dist[i]>0.:return
 a=codes[geom[i][0]];b=codes[geom[i][1]];w=world[i]
 if (a==41 and b==42) or (b==41 and a==42):wp.atomic_max(reduced,w,25,1.)
 if a==40 or b==40:
    t=b if a==40 else a
    if t>=0 and t<24:
     f=contact_force_fn(cone,frame,friction,dim,addr,ef,nj,nat,w,i,False)
     wp.atomic_add(reduced,w,24,wp.max(f[0],0.));wp.atomic_max(reduced,w,t,1.)
g=episode('scrape',gpu=True);meta=next(g);m=g.gi_frame.f_locals['mm'];m.opt.timestep=1/240;d=mj.MjData(m);mj.mj_forward(m,d)
codes=[]
for i in range(m.ngeom):
 name=m.geom(i).name;tag=re.sub(r'_\d+$','',name.split('/')[-1]);match=re.fullmatch(r'coat_(\d)_(\d)',tag)
 codes.append(int(match[1])*8+int(match[2]) if match else {'edge':40,'grip':41}.get(tag,42 if '/link_' in name else -1))
assert sum(c==40 for c in codes)==1 and len(set(c for c in codes if 0<=c<24))==24
B=int(os.getenv('BATCH','48'));N=720;ni=27;import newton
b=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(b);b.replicate(g.gi_frame.f_locals['b'],world_count=B,spacing=(0.,0.,0.))
M=b.finalize(device='cuda:0');solver=newton.solvers.SolverMuJoCo(M,use_mujoco_cpu=False,nconmax=4096,njmax=2048,iterations=40)
wm=solver.mjw_model;wd=solver.mjw_data;m=solver.mj_model
s0=M.state();s1=M.state();ctrl=M.control();newton.eval_fk(M,M.joint_q,M.joint_qd,s0);ctrlbase=ctrl.joint_target_q.numpy().reshape(B,-1).copy()
buf=wp.zeros((B,26),dtype=float);co=wp.array(codes,dtype=int)
graphs=[]
for sin,sout in [(s0,s1),(s1,s0)]:
 with wp.ScopedCapture() as cap:
    sin.clear_forces();solver.step(sin,sout,ctrl,None,1/240);buf.zero_();c=wd.contact
    wp.launch(sensor,dim=min(wd.naconmax,B*200),inputs=[c.geom,c.dist,c.worldid,wd.nacon,co,c.frame,c.friction,c.dim,c.efc_address,wd.efc.force,wd.njmax,int(m.opt.cone),buf])
 graphs.append(cap.graph)

from scipy.spatial.transform import Rotation as R
ref=np.load(OUT/'scrape_gpu_search/candidate_002.npz')['actions']
alt=np.load(OUT/'scrape_search/candidate_003.npz')['actions'];files=list(sorted((OUT/'scrape_gpu_search').glob('candidate*.npz')))+list(sorted((OUT/'scrape_search').glob('candidate*.npz')))
A=np.stack([np.load(f)['actions'] for f in files]);mount=R.from_euler('x',90,degrees=True)
N=455;lim=m.jnt_range[:27].T;rng=np.random.default_rng(48);root=OUT/'scrape_transport';root.mkdir(exist_ok=True)
lo=np.array([0,-4,-4,-4,-.45,-.55,-.65,-.55,-.4,0,0,0, -40,-25,-30,-.015,-.012,-.018, .005,-.008]);hi=np.array([len(A)-.001,4,4,4,.45,.55,.65,.55,.4,.85,1.15,1.25,35,25,30,.014,.012,.014,.045,.008])
def smooth(v):
 h=np.clip(v,0,1);return h*h*(3-2*h)
archive=[];top=-999
for it in range(125):
 p=rng.uniform(lo,hi,(B,len(lo)))
 if archive:
    k=B//5;parents=np.stack([v[1] for v in archive]);p[k:]=np.clip(parents[rng.integers(len(parents),size=B-k)]+rng.normal(size=(B-k,len(lo)))*(hi-lo)*rng.choice([.01,.035,.12,.25],(B-k,1)),lo,hi)
 action=np.zeros((N,B,27),np.float32);wp.copy(s0.joint_q,M.joint_q);wp.copy(s0.joint_qd,M.joint_qd);newton.eval_fk(M,M.joint_q,M.joint_qd,s0);wd.qacc_warmstart.zero_();wd.time.zero_()
 reward=np.zeros(B);lifted=np.zeros(B);bad=np.zeros(B);heldtime=np.zeros(B);sub=A[p[:,0].astype(int)];peak=np.zeros(B)
 for i in range(N):
    t=i/240;f=smooth((i-110)/145);h=smooth((i-292)/130)
    a=sub[:,min(i,439),:].copy();a[:,:3]+=p[:,1:4]*.001*f
    a[:,6:11]+=f*p[:,4:9];a[:,16:19]+=f*p[:,9:12]
    if i>=292:
     if i==292:
      start=ed.copy();lever=qpos[:,27:30]-[0,0,.091]-mount.apply(qpos[:,:3]);base=a[:,:3].copy();fb=np.zeros((B,3))
     rot=R.from_euler('ZYX',p[:,12:15]*h,degrees=True)
     a[:,3:6]=(mount.inv()*rot*mount*R.from_euler('XYZ',a[:,3:6])).as_euler('XYZ')
     goal=np.column_stack([-.080+p[:,15],.006+p[:,16],.044+p[:,17]])
     target=(1-h)*start+h*goal;target[:,2]+=.017*np.sin(h*np.pi)
     fb=np.clip(.989*fb+p[:,18,None]*(target-ed),-.013,.013)
     a[:,:3]=base+mount.inv().apply(target-start-(rot.apply(lever)-lever)+fb)
    action[i]=np.clip(a,*lim);ctrlbase[:,:27]=action[i];ctrl.joint_target_q.assign(ctrlbase.ravel());wp.capture_launch(graphs[i%2])
    qpos=wd.qpos.numpy();op=qpos[:,27:30];ori=R.from_quat(qpos[:,[31,32,33,30]]);ed=op+ori.apply([-.025,0,-.003]);ss=buf.numpy();held=ss[:,25]>.5
    peak=np.maximum(peak,held*(op[:,2]-.04))
    if 325<i<440:heldtime+=held/114
    if i>=390:
     distance=np.linalg.norm(ed-[-.080,.006,.038],axis=1);theta=np.arcsin(np.clip(ori.apply([1,0,0])[:,2],-1,1))
     wide=abs(ori.apply([0,1,0])[:,1]);flat=abs(ori.apply([0,1,0])[:,2]);height=np.exp(-distance/.019)
     reward+=held*(2*height+4*height*np.exp(-abs(theta-.51)/.34)+3*height*wide*np.exp(-flat/.25))/65
 score=reward+heldtime*2+np.minimum(peak/.014,1)
 j=score.argmax();archive.extend((score[l],p[l].copy()) for l in np.argsort(score)[-5:]);archive.sort(key=lambda z:-z[0]);archive=archive[:15]
 if score[j]>top:
    top=score[j];np.savez(root/f'best{it:03d}.npz',actions=action[:,j],p=p[j],score=top);print(it,top,heldtime[j],ed[j],ori[j].as_euler('ZYX',degrees=True),flush=True)
