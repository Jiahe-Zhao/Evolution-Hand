
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

def episode(task,preview=False,gpu=False):
    b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003
    from surface_objects import scene as object_scene, Score
    new=task in ("pebble","scrape");score=Score(task) if new else None
    b.add_mjcf(str(object_scene(task) if new else scene(task)),collapse_fixed_joints=False)
    joints=[str(x).split('/')[-1] for x in b.joint_label]
    for i,name in enumerate(joints):
     k=b.joint_qd_start[i]
     if 'diagnostic_wrist' in name:b.joint_target_ke[k]=1700;b.joint_target_kd[k]=42
     if name=='hinge':b.joint_target_ke[k]=0;b.joint_target_kd[k]=0
    if task=='phone':
     k=joints.index('link_1_0_to_link_1_1');b.joint_q[b.joint_q_start[k]]=.85;b.joint_target_q[b.joint_qd_start[k]]=.85
    m=b.finalize(device='cuda:0' if gpu else 'cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);c=m.control();solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=not gpu,nconmax=4096 if new else 1024,njmax=2048,iterations=40)
    names=[str(a).split('/')[-1] for a in m.body_label]; palm=names.index('link_0_0');item=names.index('item');tips=[names.index('link_%d_3'%i) for i in range(1,6)]
    base=c.joint_target_q.numpy().copy();dt=1/240
    mm=solver.mj_model;dd=solver.mj_data
    vm=mj.MjModel.from_xml_path(str(OUT/(task+'.xml')));vd=mj.MjData(vm)
    joint_map=[]
    for j,name in enumerate(joints):
     k=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)
     if k>=0:joint_map.append((vm.jnt_qposadr[k],b.joint_q_start[j],7 if vm.jnt_type[k]==0 else 1))
    renderer=mj.Renderer(vm,height=360,width=480) if preview else None;cam=mj.MjvCamera();cam.lookat[:]=[-.079,-.02,.11];cam.distance=.28;cam.azimuth=117;cam.elevation=-32
    init=s.body_q.numpy().copy();rel0=R.from_quat(init[palm,3:]).inv().apply(init[item,:3]-init[palm,:3]);normal0=R.from_quat(init[palm,3:]).inv().apply(R.from_quat(init[item,3:]).apply([0,0,1]));desired=R.from_rotvec(np.array([0,0,1])*np.pi/4).apply(normal0)
    seq=held=best=off=0;bad=False;events=[];dt=1/240
    N=1200 if task=='handle' else 720
    active=[(name,b.joint_qd_start[k]) for k,name in enumerate(joints) if name.startswith(('wrist_','diagnostic_wrist_','link_')) and mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)>=0]
    action_names=[v[0] for v in active];indices=[v[1] for v in active]
    lo,hi=vm.jnt_range[[vm.joint(k).id for k in action_names]].T.astype('f')
    a=yield dict(names=names,action_names=action_names,lo=lo,hi=hi,base=base[indices],poses=init,dt=dt,horizon=N)

    for i in range(N):
     t=i*dt
     cmd=base.copy();cmd[indices]=np.clip(a,lo,hi);c.joint_target_q.assign(cmd);s.clear_forces();solver.step(s,n,c,None,dt);s,n=n,s
     if gpu:
      import mujoco_warp as mjw
      mjw.get_data_into(dd,mm,solver.mjw_data)
     x=s.body_q.numpy();assert np.isfinite(x).all();P=R.from_quat(x[palm,3:]);O=R.from_quat(x[item,3:]);rel=P.inv().apply(x[item,:3]-x[palm,:3]);drift=np.linalg.norm(rel-rel0)
     err=np.linalg.norm(x[item,:2]-init[item,:2]-[.05,0]);ang=np.rad2deg(np.arccos(np.clip(P.inv().apply(O.apply([0,0,1]))@desired,-1,1)))
     tilt=np.rad2deg(np.arccos(np.clip(O.apply([0,0,1])[2],-1,1)));lift=x[item,2]-init[item,2]
     long_contact=False;thumb_any=False;touch=False;hit=-1;handle=False;support=False
     for con in dd.contact:
        if con.dist>0:continue
        aa,bb=[mm.geom(g).name.split('/')[-2] for g in con.geom]
        if aa in ('item','bail'):aa,bb=bb,aa
        if bb=='item' and aa.startswith('link_'):
         support=True
         if task=='handle':bad=True
         long_contact |= aa.startswith(('link_2','link_3','link_4','link_5','link_palm'))

         if aa=='link_1_3':
          thumb_any=True
          pt=O.inv().apply(con.pos-x[item,:3])

          if pt[2]>=.002:
           touch=True
           for k,y in enumerate([-.039,-.053]):
            if np.linalg.norm(pt[:2]-[[.025,.025][k],y])<=.006:hit=k
        if bb=='bail' and aa.startswith('link_'):handle=True
     hinge=float(s.joint_q.numpy()[b.joint_q_start[joints.index('hinge')]]) if 'hinge' in joints else 0
     if task=='phone':
        prev=seq
        if drift>.012 or x[item,2]<.04:bad=True
        if seq==0 and hit==0 and long_contact:seq=1
        elif seq==1:
         off=0 if thumb_any else off+1
         if off>=12:seq=2
        elif seq==2 and hit==1 and long_contact:seq=3
        if prev!=seq:events.append(dict(phase=seq,time=t,drift_mm=1000*drift))
     ok={'carry':err<=.008 and drift<=.015 and support,'stone':ang<=10 and drift<=.02 and support,'phone':seq==3 and drift<=.012 and long_contact and not bad,'handle':hinge>=np.pi/4 and lift>=.025 and tilt<=15 and handle and not bad}.get(task,False) and x[item,2]>.04
     held=held+1 if ok else 0;best=max(best,held)
     threshold={'stone':72,'phone':1}.get(task,120)
     detail=score.update(mm,dd,x,init,dict(zip(names,range(len(names)))),t) if new else {}
     frame=None
     if preview and i%8==0:
        q=s.joint_q.numpy()
        for dst,src,count in joint_map:
         vd.qpos[dst:dst+count]=q[src:src+count] if count==1 else np.r_[q[src:src+3],q[src+6],q[src+3:src+6]]
        if task=="scrape":
         for cell,travel in score.travel.items():
          f=min(travel/.003,1);vm.geom_rgba[vm.geom(cell).id,:3]=np.array([.74,.53,.32])*(1-f)+np.array([.49,.35,.23])*f
        mj.mj_forward(vm,vd);renderer.update_scene(vd,cam);frame=renderer.render().copy()
     metrics=[t,drift,err,ang,hinge,lift,tilt,seq,int(support),int(handle)]
     a=yield dict(task_metrics=detail,poses=x,metrics=list(map(float,metrics)),frame=frame,events=events.copy(),best_s=best*dt,success=detail.get("success",bool(best>=threshold)),timeout=i+1==N)
