
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
src=OUT/'scrape_gpu_search/candidate_002.npz';A0=np.load(src)['actions'];mount=R.from_euler('x',90,degrees=True);root=OUT/os.getenv('DEST','scrape_detour');root.mkdir(exist_ok=True)
rng=np.random.default_rng(int(os.getenv('SEED','18')));lim=m.jnt_range[:27].T
def ramp(t):return np.clip(t,0,1)
def smooth(t):x=ramp(t);return x*x*(3-2*x)
# delta orientation world y, lateral and height correction, force response, grip tweaks and transport time
lo=np.array([-35,-8,-10,-15,-8,-.22,-.35,-.30,.008,.002,0,.0,-.3,-.3,0]);hi=np.array([25,8,10,12,7,.22,.35,.30,.08,.05,.00016,.8,.3,.3,1])
lo=np.r_[lo,-.012,-.012,-.012,-15,-.005];hi=np.r_[hi,.012,.012,.012,15,.005]
lo=np.r_[lo,-.023,-.023,-.018,-.018,-.006,-.006,-20,-35,-18];hi=np.r_[hi,.023,.023,.018,.018,.006,.006,20,35,18]
from raw_contacts import collect
tool=wp.array([int('/item/' in m.geom(i).name) for i in range(m.ngeom)],dtype=int)
raw=wp.zeros((N,B,32,20),dtype=float);count=wp.zeros((N,B),dtype=int)
fixed=np.load(OUT/'scrape_search_recorded/best066.npz')['p'];K=len(fixed)
lo=np.r_[lo, -30,-25,-15,-.65,-.65,-.65,-.6,-.6,-45,-40,-.01,-.012,-.012,-.5,-.5,-.5,-.5,540,15,0, -13,-10];hi=np.r_[hi,30,25,15,.65,.65,.65,.6,.6,45,40,.01,.012,.012,.5,.5,.5,.5,645,100,1,13,10]
Q=len(lo);ref=np.load(OUT/'scrape_branch_search/best098.npz')['p']
nk=14;naxis=7;scale=np.array([.025,.030,.012,.55,.55,.60,.45])
lo=np.r_[lo,np.tile(-scale,nk)];hi=np.r_[hi,np.tile(scale,nk)]
L=len(lo);star=np.load(OUT/'scrape_spline_search/best104.npz')['p']
vl=np.array([-.025,-.035,-.012,-.65,-.60,-.8,-.65,-.65,-.8,0,-.01,-.015,-.008,-.5,-.6,-.55]);vh=np.array([.025,.035,.012,.65,.60,.8,.65,.65,1.3,1,.01,.015,.008,.5,.6,.55]);lo=np.r_[lo,vl];hi=np.r_[hi,vh]
split=int(os.getenv('SPLIT','530'));delay=int(os.getenv('DELAY','58'))
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
 for real_i in range(N):
    i=real_i if real_i<split else max(split,real_i-delay)
    base=A0[min(i,439)].copy();a=np.broadcast_to(base,(B,27)).copy()
    if i>=300:
     # Move the gripped object to the board; terminate wrist retreat at a lower height.
     if i==300:
      qbase=A0[-1];end=mount*R.from_euler('XYZ',qbase[3:6]);lever=qpos[:,27:30]-np.array([0,0,.091])-mount.apply(qpos[:,:3]);start=ed.copy()
     t=smooth((i-300)/135);u=smooth((i-417)/58);x=ramp((i-469)/97)-ramp((i-598)/103);y=smooth((i-566)/32)
     stage1=smooth((i-445)/45)*(1-smooth((i-560)/40));stage2=smooth((i-560)/40)
     branch=smooth((i-p[:,K+17])/p[:,K+18]);wave=np.sin(np.pi*2*(i-470)/np.maximum(45,p[:,K+18]))*smooth((i-470)/25)*(1-smooth((i-590)/35))
     angle=np.deg2rad(p[:,0]+p[:,18]*u+p[:,26]*stage2+p[:,K+8]*branch)*t
     ro=R.from_rotvec(angle[:,None]*[0,1,0]);a[:,3:6]=(mount.inv()*ro*end).as_euler('XYZ')
     a[:,10]+=p[:,5]*t;a[:,13]+=p[:,6]*t;a[:,14]+=p[:,7]*t
     a[:,[6,10,13,14,12]]+=p[:,K+3:K+8]*branch[:,None];a[:,[9,8,16,17]]+=p[:,K+13:K+17]*branch[:,None];a[:,5]+=np.deg2rad(p[:,K+9])*branch
     a[:,6]+=p[:,12]*t;a[:,12]+=p[:,13]*t
     reverse=p[:,14]>.5;goal=np.column_stack([np.where(reverse,-.078,-.134)+np.where(reverse,-1,1)*(.056+p[:,15])*x+p[:,1]/1000,.0004+p[:,16]+(.014+p[:,17])*y+p[:,2]/1000,.035+(p[:,3]/1000)+(.018+p[:,4]/1000)*(1-u)+p[:,19]*y])
     goal[:,0]+=stage1*p[:,20]+stage2*p[:,21];goal[:,1]+=stage1*p[:,22]+stage2*p[:,23];goal[:,2]+=stage1*p[:,24]+stage2*p[:,25]
     goal[:,0]+=p[:,K]*.001*branch;goal[:,1]+=p[:,K+1]*.001*branch+p[:,K+20]*.001*wave;goal[:,2]+=p[:,K+2]*.001*branch+p[:,K+21]*.0004*wave
     target=(1-t)*start+t*goal
     # Feed forward wrist displacement about the commanded rotation center.
     dp=target-start-(ro.apply(lever)-lever)
     if last is not None:
      feedback=np.clip(feedback*.987+p[:,8,None]*(target-ed),-.016,.016)
      if i>460:fctrl=np.clip(fctrl+p[:,10]*(sensors[:,24]-.95),-.010,.005)
     dp+=feedback;dp[:,2]+=fctrl
     a[:,:3]=A0[300,:3]+mount.inv().apply(dp)
     if i>=425:
      f=np.clip((i-425)/21,0,nk-.00001);j=int(f);w=smooth(f-j)
      knobs=p[:,Q:L].reshape(B,nk,naxis)
      patch=knobs[:,j,:]*w+(knobs[:,j-1,:]*(1-w) if j else 0)
      a[:,0]+=patch[:,0];a[:,2]-=patch[:,1];a[:,1]+=patch[:,2]
      a[:,3]+=patch[:,3];a[:,5]+=patch[:,4];a[:,13]+=patch[:,5];a[:,10]+=patch[:,6]
      if real_i>=split:
       phase=np.clip((real_i-split)/delay,0,1)
       wave=np.sin(np.pi*phase)**1.5
       persistent=smooth(phase)*(1-smooth((real_i-split-delay)/40))
       rr=R.from_rotvec(wave*p[:,L+3:L+6]);lev=op-[0,0,.091]-mount.apply(qpos[:,:3])+p[:,L+8,None]*ori.apply([-.025,0,-.003])
       a[:,3:6]=(mount.inv()*rr*mount*R.from_euler('XYZ',a[:,3:6])).as_euler('XYZ')
       a[:,:3]+=mount.inv().apply(wave*p[:,L:L+3]+lev-rr.apply(lev)+persistent*p[:,L+10:L+13])
       a[:,10]+=p[:,L+6]*wave+p[:,L+13]*persistent;a[:,13]+=p[:,L+7]*wave+p[:,L+14]*persistent;a[:,14]+=p[:,L+15]*persistent
    a=np.clip(a,lim[0],lim[1]);actions[real_i]=a;ctrlbase[:,:27]=a;ctrl.joint_target_q.assign(ctrlbase.ravel());wp.capture_launch(graphs[real_i%2])
    c=wd.contact
    wp.launch(collect,dim=B*200,inputs=[real_i,c.geom,c.dist,c.worldid,wd.nacon,tool,c.dim,c.efc_address,c.friction,wd.efc.force,int(m.opt.cone),count,raw])
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
 obj=5*cells+1.7*np.minimum(dist/.003,1).sum(1)+2*near+progress+grip+np.minimum(maxlift/.012,1)*2
 return obj,cells,actions
for it in range(int(os.getenv('ROUNDS','70'))):
 p=rng.uniform(lo,hi,(B,len(lo)));p[:,:L]=star
 if archive:
    k=B//5;parents=np.stack([z[1] for z in sorted(archive,key=lambda z:-z[0])[:22]]);p[k:]=parents[rng.integers(len(parents),size=B-k)];p[k:,L:]+=rng.normal(size=p[k:,L:].shape)*(hi[L:]-lo[L:])*rng.choice([.009,.023,.06,.16],(B-k,1));p=np.clip(p,lo,hi)
 cost,cells,act=roll(p);
 if it==0 and not files:archive=[]
 k=cost.argmax();archive.extend((float(cost[j]),p[j].copy()) for j in np.argsort(cost)[-5:]);archive.sort(key=lambda z:-z[0]);archive=archive[:14]
 if cost[k]>top:
    top=cost[k];np.savez(root/f'best{it+offset:03d}.npz',score=top,p=p[k],actions=act[:,k],cells=cells[k]);np.savez_compressed(root/f'record{it+offset:03d}.npz',qpos=np.stack(qhist)[:,k],sensors=np.stack(shist)[:,k],contacts=raw.numpy()[:,k],counts=count.numpy()[:,k],geom_names=[m.geom(j).name for j in range(m.ngeom)],lane=k,batch=B,cone=int(m.opt.cone));print(it,'best',top,'tiles',cells[k],flush=True)
 if cells[k]>=20:break
