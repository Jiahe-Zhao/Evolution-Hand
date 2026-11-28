
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os
os.environ.setdefault('MUJOCO_GL','egl')
from build import *
import sys,json,warp as wp,newton
from scipy.spatial.transform import Rotation as R
task=sys.argv[1];negative='negative' in sys.argv;tag=task+('_negative' if negative else '')
tag+=os.environ.get('TRIAL','');preview=os.environ.get('NO_VIDEO')!='1'
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003
b.add_mjcf(str(scene(task)),collapse_fixed_joints=False)
joints=[str(x).split('/')[-1] for x in b.joint_label]
for i,name in enumerate(joints):
 k=b.joint_qd_start[i]
 if 'diagnostic_wrist' in name:b.joint_target_ke[k]=1700;b.joint_target_kd[k]=42
 if name=='hinge':b.joint_target_ke[k]=0;b.joint_target_kd[k]=0
if task=='phone':
 k=joints.index('link_1_0_to_link_1_1');b.joint_q[b.joint_q_start[k]]=.85;b.joint_target_q[b.joint_qd_start[k]]=.85
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);c=m.control();solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40)
names=[str(a).split('/')[-1] for a in m.body_label]; palm=names.index('link_0_0');item=names.index('item');tips=[names.index('link_%d_3'%i) for i in range(1,6)]
base=c.joint_target_q.numpy().copy();trajectory=[];actions=[];frames=[];stats=[];dt=1/240
mm=solver.mj_model;dd=solver.mj_data
vm=mj.MjModel.from_xml_path(str(OUT/(task+'.xml')));vd=mj.MjData(vm)
joint_map=[(int(vm.jnt_qposadr[k]),int(mm.jnt_qposadr[next(j for j in range(mm.njnt) if mm.joint(j).name.endswith(vm.joint(k).name) or ('/'+vm.joint(k).name+'_') in mm.joint(j).name)]),7 if vm.jnt_type[k]==0 else 1) for k in range(vm.njnt)] if False else []
for j,name in enumerate(joints):
 k=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)
 if k>=0:joint_map.append((vm.jnt_qposadr[k],b.joint_q_start[j],7 if vm.jnt_type[k]==0 else 1))
renderer=mj.Renderer(vm,height=360,width=480) if preview else None;cam=mj.MjvCamera();cam.lookat[:]=[-.079,-.02,.11];cam.distance=.28;cam.azimuth=117;cam.elevation=-32
def target(a,name,value):
 if name in joints:a[b.joint_qd_start[joints.index(name)]]=value
init=s.body_q.numpy().copy();rel0=R.from_quat(init[palm,3:]).inv().apply(init[item,:3]-init[palm,:3]);normal0=R.from_quat(init[palm,3:]).inv().apply(R.from_quat(init[item,3:]).apply([0,0,1]));desired=R.from_rotvec(np.array([0,0,1])*np.pi/4).apply(normal0)
def ramp(v):return np.clip(v,0,1)
seq=0;last=False;held=0;best=0;bad=False
N=1200 if task=='handle' else 720
for i in range(N):
 t=i*dt;a=base.copy()
 if not negative:
    if task=='carry':target(a,'diagnostic_wrist_x',.050*ramp((t-.4)/1.3))
    if task=='stone':
     for j in [2,3,4,5]:target(a,f'link_{j}_0_to_link_{j}_1',float(os.environ.get('BEND'+str(j),'.98'))*ramp((t-.45)/.9))
    if task=='phone':
     target(a,'link_1_thumb_spread_joint',.16*ramp((t-1.45)/.5))
     target(a,'link_1_0_to_link_1_1',.85*(1-ramp((t-.35)/.50)+ramp((t-1.0)/.40)-ramp((t-2.)/.5)))
    if task=='handle':
     target(a,'diagnostic_wrist_y',.119*(.5-.5*np.cos(np.pi*ramp((t-.45)/1.95))));target(a,'diagnostic_wrist_x',-.050*(.5-.5*np.cos(np.pi*ramp((t-.55)/1.8))))
     pass
 c.joint_target_q.assign(a);s.clear_forces();solver.step(s,n,c,None,dt);s,n=n,s
 x=s.body_q.numpy();assert np.isfinite(x).all();P=R.from_quat(x[palm,3:]);O=R.from_quat(x[item,3:]);rel=P.inv().apply(x[item,:3]-x[palm,:3]);drift=np.linalg.norm(rel-rel0)
 err=np.linalg.norm(x[item,:2]-init[item,:2]-[.05,0]);ang=np.rad2deg(np.arccos(np.clip(P.inv().apply(O.apply([0,0,1]))@desired,-1,1)))
 tilt=np.rad2deg(np.arccos(np.clip(O.apply([0,0,1])[2],-1,1)));lift=x[item,2]-init[item,2]
 pairs=[];touch=False;hit=-1;handle=False;support=False
 for con in dd.contact:
    if con.dist>0:continue
    aa,bb=[mm.geom(g).name.split('/')[-2] for g in con.geom]
    if aa in ('item','bail'):aa,bb=bb,aa
    if bb=='item' and aa.startswith('link_'):
     support=True
     if task=='handle' and not bad:print('BODY',t,aa,con.dist);bad=True
     if task=='phone' and aa.startswith('link_1') and i%30==0:print('contact',round(t,2),aa,O.inv().apply(con.pos-x[item,:3]).round(4))
     if aa=='link_1_3':
      pt=O.inv().apply(con.pos-x[item,:3])
      if i%20==0: print('THUMB',round(t,2),pt.round(4))
      if pt[2]>=.002:
       touch=True
       for k,y in enumerate([-.039,-.053]):
        if np.linalg.norm(pt[:2]-[[.025,.025][k],y])<=.006:hit=k
    if bb=='bail' and aa.startswith('link_'):handle=True
 hinge=float(s.joint_q.numpy()[b.joint_q_start[joints.index('hinge')]]) if 'hinge' in joints else 0
 if task=='phone':
    if seq==0 and hit==0:seq=1
    elif seq==1 and not touch:seq=2
    elif seq==2 and hit==1:seq=3
 ok={'carry':err<=.008 and drift<=.015 and support,'stone':ang<=10 and drift<=.02 and support,'phone':seq==3 and drift<=.012 and support,'handle':hinge>=np.pi/4 and lift>=.025 and tilt<=15 and handle and not bad}[task] and x[item,2]>.04
 held=held+1 if ok else 0;best=max(best,held)
 if i%8==0:
    trajectory.append(x.copy());actions.append(a.copy());stats.append([t,drift,err,ang,hinge,lift,tilt,seq,int(support),int(handle)])
    if not preview:continue
    q=s.joint_q.numpy()
    for dst,src,count in joint_map:
     vd.qpos[dst:dst+count]=q[src:src+count] if count==1 else np.r_[q[src:src+3],q[src+6],q[src+3:src+6]]
    mj.mj_forward(vm,vd);renderer.update_scene(vd,cam);frames.append(renderer.render().copy())
import subprocess
if preview:subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','480x360','-framerate','30','-i','-','-pix_fmt','yuv420p',str(OUT/(tag+'.mp4'))],input=np.asarray(frames).tobytes(),check=True)
np.savez_compressed(OUT/(tag+'.npz'),poses=trajectory,names=names,actions=actions,metrics=stats)
result=dict(task=task,negative=negative,backend='Newton SolverMuJoCo CPU',steps=N,dt=dt,success=bool(best>=({'stone':72,'phone':1}.get(task,120))),best_s=best*dt,last=list(map(float,stats[-1])),columns='time drift goal_xy_error face_error_deg hinge_rad lift tilt_deg phone_phase hand_contact bail_contact'.split())
(OUT/(tag+'.json')).write_text(json.dumps(result,indent=2));print(result)
