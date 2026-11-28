
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
src=OUT/'scrape_gpu_search/candidate_002.npz';A0=np.load(src)['actions'];mount=R.from_euler('x',90,degrees=True);root=OUT/os.getenv('DEST','scrape_beam_search');root.mkdir(exist_ok=True)
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
Q=len(lo);fixed_base=np.load(OUT/'scrape_branch_search/best098.npz')['p'];start_tick=int(os.getenv('START','468'));span=int(os.getenv('SPAN','20'));blocks=(N-start_tick+span-1)//span;dim=8
archive=[];files=list(root.glob('best*.npz'))
for f in files:
 rec=np.load(f);archive.append((float(rec['score']),np.pad(rec['p'],(0,len(lo)-len(rec['p'])))))
top=-999;offset=max((int(f.stem[4:]) for f in files),default=-1)+1
def roll(seq,stop):
 p=np.broadcast_to(fixed_base,(B,Q));anchor=None;lastk=-1;entry=None;validhist=[]
 global qhist,shist
 qhist=[];shist=[];count.zero_()
 wp.copy(s0.joint_q,M.joint_q);wp.copy(s0.joint_qd,M.joint_qd);newton.eval_fk(M,M.joint_q,M.joint_qd,s0)
 wd.qacc_warmstart.zero_();wd.time.zero_()
 actions=np.zeros((N,B,27),np.float32);dist=np.zeros((B,24));near=np.zeros(B);progress=np.zeros(B);maxlift=np.zeros(B);grip=np.zeros(B);fctrl=np.zeros(B);feedback=np.zeros((B,3));last=None
 for i in range(stop):
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
     if i>=start_tick:
      k=(i-start_tick)//span;alpha=((i-start_tick)%span+1)/span
      if lastk!=k:
       anchor=actions[i-1].astype(float).copy();entry=a.copy();lastk=k
      vv=seq[:,k];anchorplus=anchor.copy()
      # Follow baseline or hold the last pose, then apply a new local wrist / finger motion.
      anchorplus+=(a-entry)*vv[:,7,None]
      anchorplus[:,0]+=alpha*vv[:,0];anchorplus[:,2]-=alpha*vv[:,1];anchorplus[:,1]+=alpha*vv[:,2]
      anchorplus[:,3]+=alpha*vv[:,3];anchorplus[:,5]+=alpha*vv[:,4]
      anchorplus[:,10]+=alpha*vv[:,5];anchorplus[:,13]+=alpha*vv[:,6]
      a=anchorplus
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
    validhist.append(valid)
 cells=(dist>=.003).sum(1)
 recent=np.array(validhist[-10:]).mean(0)
 quality=4*recent+3*held+np.minimum(maxlift/.014,1)+np.exp(-np.abs(ang-30)/15)*2
 partial=np.minimum(dist/.003,1).sum(1)
 # Prefer progress, preserve a handle on the implement for the next segment.
 score=3*cells+1.2*partial+quality
 return score,cells,actions,dist,ed,quality

pool=[np.zeros((blocks,dim))];directions=np.array([.014,.018,.005,.22,.22,.24,.23,1.])
best=-999
for k in range(blocks):
 stop=min(N,start_tick+(k+1)*span);records=[]
 for trial in range(int(os.getenv('TRIES','8'))):
    parent=rng.integers(len(pool),size=B);seq=np.stack([pool[v].copy() for v in parent])
    if trial>1 and records:
     for j in range(B//2):
      seq[j]=records[rng.integers(min(14,len(records)))][1].copy()
    mod=rng.normal(size=(B,dim))*directions*rng.choice([.12,.45,.8,1.2],size=(B,1));seq[:,k]+=mod;seq[:,k,7]=np.clip(seq[:,k,7]+.5,0,1)
    seq[0]=pool[0];seq[0,k,7]=1
    cost,cells,actions,distp,edge,quality=roll(seq,stop);j=cost.argmax()
    for l in np.argsort(cost)[-16:]:
     mask=tuple(np.where(distp[l]>=.003)[0]);records.append((cost[l],seq[l].copy(),(mask,tuple(np.round(edge[l,:2]/.008).astype(int)))))
    records.sort(key=lambda z:-z[0]);ss=set();records=[v for v in records if not(v[2] in ss or ss.add(v[2]))][:45]
    # Every candidate above ran from the original reset; no object state cloning.
    if cells.max()>=20 or (k==blocks-1 and cost[j]>best):
     if cells.max()>=20:j=np.argmax(cells)
     tag=k*100+trial;best=cost[j]
     np.savez(root/f'best{tag:03d}.npz',actions=actions[:stop,j],cells=cells[j],seq=seq[j]);np.savez_compressed(root/f'record{tag:03d}.npz',qpos=np.stack(qhist)[:,j],sensors=np.stack(shist)[:,j],contacts=raw.numpy()[:stop,j],counts=count.numpy()[:stop,j],geom_names=[m.geom(h).name for h in range(m.ngeom)],lane=j,batch=B,cone=int(m.opt.cone))
     if cells[j]>=20:print('PASS',k,trial,cells[j],flush=True);sys.exit(0)
 pool=[v[1] for v in records[:18]];print('block',k,stop,'best',len(records[0][2][0]),'max',max(len(v[2][0]) for v in records),'edge',records[0][2][1],flush=True)
