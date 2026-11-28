
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os,sys,json,subprocess
from env import episode,OUT,mj,np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
task=sys.argv[1];gpu='--gpu' in sys.argv;video='--video' in sys.argv
m=mj.MjModel.from_xml_path(str(OUT/(task+'.xml')));d=mj.MjData(m)
ids=[next(i for i,b in enumerate(m.geom_bodyid) if m.body(b).name==f'link_{j}_3') for j in (1,2)]
gs=[i for i,b in enumerate(m.geom_bodyid) if m.body(b).name.startswith('link_') and i not in ids]
center=np.array([-.093,.003,.032] if task=='pebble' else [-.080,-.042,.041])
phi=float(os.getenv('PHI','90' if task=='pebble' else '101.46'))*np.pi/180
line=np.array([np.cos(phi),np.sin(phi),0])
def fk(q):
 d.qpos[:15]=q;mj.mj_kinematics(m,d)
 ax=d.geom_xmat[ids].reshape(2,3,3)[:,:,2]
 return d.geom_xpos[ids]+.007*ax,ax
lo,hi=m.jnt_range[:15].T
def plan(gap,seed,first):
 target=center+np.array([line,-line])*gap
 def cost(q):
    tip,ax=fk(q);return np.r_[((tip-target)*1000).ravel(),((ax-(np.array([-line,line])*.65+[0,0,-.76])).ravel()*18),np.minimum(d.geom_xpos[gs,2]-center[2]-.007,0)*480,(q-seed)*(.25 if first else 2)]
 return least_squares(cost,seed,bounds=(lo,hi),max_nfev=100)
tag=task+os.getenv('TAG','')
file=OUT/('opposed_'+task+'_'+str(round(phi,3))+'.npz')
if not file.exists():
 rng=np.random.default_rng(79);best=None
 for i in range(6):
    seed=np.r_[0,.03,0,rng.uniform(-1.3,1.3,3),rng.uniform(.05,.8,5),rng.uniform(-.2,.2),rng.uniform(.25,1.1,3)];r=plan(.012,seed,True)
    if best is None or r.cost<best.cost:best=r
    if r.cost<.15:break
 q=best.x;arr=[]
 for gap in np.linspace(.022,.002,21):
    q=plan(gap,q,False).x;arr.append(q)
 np.savez(file,arr=arr)
arr=np.load(file)['arr']
def getq(gap):
 u=np.clip((.022-gap)*1000,0,19.999);k=int(u);return arr[k]*(1-u+k)+arr[k+1]*(u-k)
g=episode(task,video,gpu);meta=next(g);qbase=meta['base'];order=[meta['action_names'].index(m.joint(i).name) for i in range(15)]
def ramp(t):return np.clip(t,0,1)
hist=[];frames=[];poses=[];cmd=[];names=meta['names'];ix=names.index('item')
palm=names.index('link_0_0')
bids=[m.geom_bodyid[g] for g in ids];liveids=[names.index(m.body(b).name) for b in bids]
fk(np.zeros(15));loc=[d.xmat[b].reshape(3,3).T@(pt-d.xpos[b]) for b,pt in zip(bids,fk(np.zeros(15))[0])]
corr=np.zeros(15);errlog=[]
for i in range(meta['horizon']):
 t=i/240;gap=.022-(.022-float(os.getenv('GAP','.006')))*ramp((t-1.25)/.42)
 q=getq(gap);q[1]+=float(os.getenv('DZ','0'))+.055*(1-ramp((t-.75)/.5))+.034*ramp((t-1.72)/.48)
 if i and t>1.3:
    off=poses[-1][ix,:2]-center[:2];q[0]+=np.clip(off[0],-.022,.022);q[2]-=np.clip(off[1],-.022,.022)
 a=qbase.copy();a[order]=q*ramp(t/.68);a[order[1]]+=.06*np.sin(np.pi*ramp(t/.68))
 if i and t>.95 and os.getenv('FB','1')=='1':
    obs=poses[-1];real=obs[liveids,:3]+R.from_quat(obs[liveids,3:]).apply(loc)
    des=fk(q)[0];err=des-real
    # Correct dynamic tracking error in task space, retaining the reference posture.
    jac=[]
    for j in range(15):
     v=q.copy();v[j]+=.0001;jac.append(((fk(v)[0]-des)/.0001).ravel())
    J=np.array(jac).T;cost=np.r_[np.ones(3)*.045,np.ones(3)*1.5,np.ones(9)];M=J*cost
    dq=cost*(M.T@np.linalg.solve(M@M.T+.00004*np.eye(6),err.ravel()))
    corr=np.clip(.994*corr+.16*dq,-np.r_[[.018]*3,[.5]*12],np.r_[[.018]*3,[.5]*12])
    a[order]+=corr
    errlog.append([i,real.tolist(),des.tolist()])
 z=g.send(a);h=z['task_metrics'];h['t']=t;h['pos']=z['poses'][ix,:3].tolist()
 hist.append(h);poses.append(z['poses'].copy());cmd.append(a)
 if z['frame'] is not None:frames.append(z['frame'])
if video:
 sp=subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r','30','-i','-','-pix_fmt','yuv420p',str(OUT/(tag+'_attempt.mp4'))],input=np.asarray(frames).tobytes(),check=True)
(OUT/(tag+'_err.json')).write_text(json.dumps(errlog));(OUT/(tag+'_attempt.json')).write_text(json.dumps(hist));np.savez_compressed(OUT/(tag+'_attempt.npz'),actions=cmd,poses=poses,names=names)
print(tag,h,flush=True)
