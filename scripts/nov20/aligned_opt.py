
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
import glob
files0=list(sorted((OUT/'scrape_gpu_search').glob('candidate*.npz')))+list(sorted((OUT/'scrape_search').glob('candidate*.npz')))
trans=np.load(OUT/os.environ['PICK']);pp=trans['p'];A=np.load(files0[int(pp[0])])['actions']
mount=R.from_euler('x',90,degrees=True);root=OUT/os.getenv('DEST','scrape_aligned');root.mkdir(exist_ok=True)
rng=np.random.default_rng(int(os.getenv('SEED','18')));lim=m.jnt_range[:27].T
def ramp(t):return np.clip(t,0,1)
def smooth(t):h=ramp(t);return h*h*(3-2*h)
lo=np.array([-.010,.018,-.072,.000,-.016,-.012,.001,.00,0,10,-.8,-.6,-.5,-.5,-.5,-.4,0,0,40,-.012,-.012])
hi=np.array([.012,.050,-.040,.018,.016,.012,.11,.6,.00022,46,1.1,1.3,.5,.5,.5,.4,.6,1,170,.012,.012])
from raw_contacts import collect
tool=wp.array([int('/item/' in m.geom(i).name) for i in range(m.ngeom)],dtype=int)
raw=wp.zeros((N,B,32,20),dtype=float);count=wp.zeros((N,B),dtype=int)
archive=[];files=list(root.glob('best*.npz'))
for f in files:
 rec=np.load(f);archive.append((float(rec['score']),np.pad(rec['p'],(0,len(lo)-len(rec['p'])))))
top=-999;offset=max((int(f.stem[4:]) for f in files),default=-1)+1
def roll(p):
 global qhist,shist
 qhist=[];shist=[];count.zero_()
 wp.copy(s0.joint_q,M.joint_q);wp.copy(s0.joint_qd,M.joint_qd);newton.eval_fk(M,M.joint_q,M.joint_qd,s0)
 wd.qacc_warmstart.zero_();wd.time.zero_()
 actions=np.zeros((N,B,27),np.float32);dist=np.zeros((B,24));near=np.zeros(B);progress=np.zeros(B);maxlift=np.zeros(B);grip=np.zeros(B);fctrl=np.zeros(B);feedback=np.zeros((B,3));last=None
 for i in range(N):
    if i<448:
     f=smooth((i-110)/145);h=smooth((i-292)/130)
     a=np.tile(A[min(i,439)],(B,1));a[:,:3]+=pp[1:4]*.001*f;a[:,6:11]+=f*pp[4:9];a[:,16:19]+=f*pp[9:12]
     if i>=292:
      if i==292:start=ed.copy();lever=qpos[:,27:30]-[0,0,.091]-mount.apply(qpos[:,:3]);base=a[:,:3].copy();fb=np.zeros((B,3))
      rot=R.from_euler('ZYX',pp[12:15]*h,degrees=True);a[:,3:6]=(mount.inv()*rot*mount*R.from_euler('XYZ',a[:,3:6])).as_euler('XYZ')
      goal=np.array([-.080+pp[15],.006+pp[16],.044+pp[17]])
      target=(1-h)*start+h*goal;target[:,2]+=.017*np.sin(h*np.pi)
      fb=np.clip(.989*fb+pp[18]*(target-ed),-.013,.013)
      a[:,:3]=base+mount.inv().apply(target-start-(rot.apply(lever)-lever)+fb)
    else:
     if i==448:
      ad=actions[i-1].copy();start=ed.copy();lever=ed-[0,0,.091]-mount.apply(qpos[:,:3]);end=mount*R.from_euler('XYZ',ad[:,3:6]);ori0=ori
      x0=ori.apply([1,0,0]);yaw=np.arctan2(x0[:,1],x0[:,0]);pitch=np.deg2rad(p[:,9]);xd=np.column_stack([np.cos(yaw)*np.cos(pitch),np.sin(yaw)*np.cos(pitch),np.sin(pitch)])
      yd=np.column_stack([-np.sin(yaw),np.cos(yaw),np.zeros(B)]);yd*=np.sign((ori.apply([0,1,0])*yd).sum(1))[:,None]
      gd=R.from_matrix(np.stack([xd,yd,np.cross(xd,yd)],axis=2));turn=(gd*ori.inv()).as_rotvec();feedback=np.zeros((B,3));fctrl=np.zeros(B)
     t=smooth((i-448)/45);u=ramp((i-460)/180);late=smooth((i-627)/45);rot=R.from_rotvec(turn*(p[:,10]*t)[:,None]);a=ad.copy();a[:,3:6]=(mount.inv()*rot*end).as_euler('XYZ')
     a[:,10]+=p[:,12]*t;a[:,13]+=p[:,13]*t;a[:,6]+=p[:,14]*t;a[:,14]+=p[:,15]*t
     target=start.copy();target[:,0]+=p[:,0]*t+p[:,2]*u+late*p[:,1]
     target[:,1]+=(.008-start[:,1])*t+p[:,3]*late+p[:,4]*t*np.sin((i-448)*2*np.pi/p[:,18])+p[:,19]*t
     target[:,2]+=(.034-start[:,2]+p[:,5])*t+p[:,20]*late
     feedback=np.clip(.990*feedback+p[:,6,None]*(target-ed),-.022,.022);fctrl=np.clip(fctrl+p[:,8]*(sensors[:,24]-1.25),-.010,.009)
     dp=target-start-(rot.apply(lever)-lever)+feedback+(target-ed)*p[:,7,None];dp[:,2]+=fctrl;a[:,:3]=ad[:,:3]+mount.inv().apply(dp)
    a=np.clip(a,lim[0],lim[1]);actions[i]=a;ctrlbase[:,:27]=a;ctrl.joint_target_q.assign(ctrlbase.ravel());wp.capture_launch(graphs[i%2])
    c=wd.contact
    wp.launch(collect,dim=B*200,inputs=[i,c.geom,c.dist,c.worldid,wd.nacon,tool,c.dim,c.efc_address,c.friction,wd.efc.force,int(m.opt.cone),count,raw])
    qpos=wd.qpos.numpy();qhist.append(qpos);op=qpos[:,27:30];quat=qpos[:,[31,32,33,30]];ori=R.from_quat(quat);ed=op+ori.apply([-.025,0,-.003]);ang=np.degrees(np.arcsin(np.clip(ori.apply([1,0,0])[:,2],-1,1)))
    sensors=buf.numpy();shist.append(sensors);v=np.zeros(B) if last is None else np.linalg.norm(ed[:,:2]-last[:,:2],axis=1)*240;last=ed.copy();force=sensors[:,24];held=sensors[:,25]>.5
    maxlift=np.maximum(maxlift,(op[:,2]-.040)*held)
    valid=held&(ang>=15)&(ang<=45)&(force>=.3)&(force<=3)&(v>=.005)&(v<=.2)
    dist+=sensors[:,:24]*(valid*v/240)[:,None]
    if i>450:
     gd=np.linalg.norm(ed-target,axis=1)
     near+=np.exp(-gd/.018)*held/240;grip+=held/240
     progress+=np.exp(-np.abs(ang-30)/20)*np.exp(-gd/.015)*held/240
 cells=(dist>=.003).sum(1)
 obj=2*cells+np.minimum(dist/.003,1).sum(1)+7*near+6*progress+grip+np.minimum(maxlift/.012,1)*2
 return obj,cells,actions
for it in range(int(os.getenv('ROUNDS','70'))):
 p=rng.uniform(lo,hi,(B,len(lo)))
 archive.sort(key=lambda z:-z[0]);archive=archive[:18]
 if archive:
    k=B//6;e=np.stack([v[1] for v in archive]);p[k:]=np.clip(e[rng.integers(len(e),size=B-k)]+rng.normal(size=(B-k,len(lo)))*(hi-lo)*rng.choice([.005,.012,.025,.055,.12],(B-k,1)),lo,hi)
 if it==0 and archive:
    for kk,(_,prevp) in enumerate(archive[:min(len(archive),B)]):p[kk]=prevp
 cost,cells,act=roll(p);
 if it==0 and not files:archive=[]
 k=cost.argmax();archive.extend((float(cost[j]),p[j].copy()) for j in np.argsort(cost)[-5:]);archive.sort(key=lambda z:-z[0]);archive=archive[:14]
 if cost[k]>top:
    top=cost[k];np.savez(root/f'best{it+offset:03d}.npz',score=top,p=p[k],actions=act[:,k],cells=cells[k]);np.savez_compressed(root/f'record{it+offset:03d}.npz',qpos=np.stack(qhist)[:,k],sensors=np.stack(shist)[:,k],contacts=raw.numpy()[:,k],counts=count.numpy()[:,k],geom_names=[m.geom(j).name for j in range(m.ngeom)],lane=k,batch=B,cone=int(m.opt.cone));print(it,'best',top,'tiles',cells[k],flush=True)
 if cells[k]>=20:break
