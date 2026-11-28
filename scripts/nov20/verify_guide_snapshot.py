"""Blunt asymmetric stone insert; dimensions SI. Fixed wood handle, open upper socket."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import OUT,E,np,mj
from surface_objects import geom
from scipy.spatial.transform import Rotation as R
import newton,sys,json,subprocess
from pathlib import Path
out=OUT/'inlay11';out.mkdir(exist_ok=True)
xy=np.array([-.102,.030]); yaw=.70; initial=.20
# Convex pentagonal footprint: one truncated corner is the directional key.
outline=np.array([[-10,-6],[10,-6],[10,1],[5,6],[-10,6]])*.001
def world(p):
 c,s=np.cos(yaw),np.sin(yaw);return xy+np.array([[c,-s],[s,c]])@p
def scene(mode):
 r=_evo_parse_xml(OUT/'pebble.xml').getroot();w=r.find('worldbody');a=r.find('asset')
 for b in list(w):
  if b.get('name','').startswith(('item','neighbor','rim','tray')):w.remove(b)
 r.remove(r.find('contact')); c=E.SubElement(r,'contact')
 def mesh(name,pts,z0,z1):
  v=[[x,y,z] for z in [z0,z1] for x,y in pts]
  E.SubElement(a,'mesh',name=name,vertex=' '.join(map(str,np.ravel(v))))
 def solid(b,name,pts,z0,z1,color,mass=None):
  mesh(name,pts,z0,z1);return geom(b,name,'mesh',(0,0,0),'.01',color,mesh=name,**({'mass':mass} if mass else {}))
 wood='.49 .30 .16 1'; stone='.49 .54 .55 1'
 geom(w,'pickup_pad','box',(-.093,.003,.024),'.017 .014 .003','.28 .34 .35 1')
 def fixed(name,pts,z0,z1):solid(w,name,[world(p) for p in pts],z0,z1,wood)
 outer=np.array([[-.043,-.012],[.024,-.012],[.024,.012],[-.043,.012]])
 geom(w,'wood_base','box',(*world([-.0095,0]),.020),'.0335 .012 .003',wood,euler=f'0 0 {yaw}')
 # Five convex rim sectors share boundaries; the socket is empty collision volume.
 inner=outline*1.12
 shell=np.array([[-.043,-.012],[.024,-.012],[.024,.005],[.017,.012],[-.043,.012]])
 def rail(name,x0,x1,y0,y1):
  geom(w,name,'box',(*world([(x0+x1)/2,(y0+y1)/2]),.026),f'{(x1-x0)/2} {(y1-y0)/2} .003',wood,euler=f'0 0 {yaw}')
 rail('wood_left',-.043,-.0112,-.012,.012);rail('wood_right',.0112,.024,-.012,.012)
 rail('wood_back',-.0112,.0112,.00672,.012);rail('wood_front',-.0112,.0112,-.012,-.00672)
 fixed('wood_key',[inner[2],inner[3],[inner[2,0],inner[3,1]]],.023,.029)
 pos=[-.093,.003,.0314];angle=initial
 if mode in ('aligned','reverse','rim'):
  pos=[*xy,.039];angle=yaw+(np.pi if mode=='reverse' else 0)
  if mode=='rim':pos[:2]=world([-.022,0])
 b=E.SubElement(w,'body',name='item',pos=' '.join(map(str,pos)),euler=f'0 0 {angle}');E.SubElement(b,'freejoint',name='free0')
 solid(b,'flake',outline,-.004,.004,stone,'.0021')
 geom(b,'sole','box',(0,0,-.0035),'.006 .004 .0005',stone,mass='.00001')
 for g in w.findall('geom')+list(w.find('body').iter('geom')):
  E.SubElement(c,'pair',geom1='flake',geom2=g.get('name'),friction='.9 .9 .003 .0001 .0001')
 for pair in list(c):E.SubElement(c,'pair',geom1='sole',geom2=pair.get('geom2'),friction='.9 .9 .003 .0001 .0001')
 p=out/(mode+'.xml');E.ElementTree(r).write(p);return p
def run(mode):
 import os
 p=scene(mode);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
 names=[k.split('/')[-1] for k in b.joint_label]
 for i,k in enumerate(names):
  if 'diagnostic_wrist' in k:
   j=b.joint_qd_start[i];b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40)
 ix=[k.split('/')[-1] for k in m.body_label].index('item')
 valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')}
 active=[b.joint_qd_start[i] for i,k in enumerate(names) if k in valid and k.startswith(('wrist_','diagnostic_wrist_','link_'))]
 if mode=='search':
  from mujoco import rollout
  mm=v.mj_model;mm.opt.timestep=1/240;d=mj.MjData(mm);mj.mj_forward(mm,d)
  spec=mj.mjtState.mjSTATE_FULLPHYSICS;s0=np.zeros(mj.mj_stateSize(mm,spec));mj.mj_getState(mm,d,s0,spec)
  j=mm.jnt_qposadr[-1];print('INFO',mm.nu,mm.nq,j,d.qpos[j:j+3],flush=True)
  seed=np.load(OUT/'positive_review/pebble_candidate_045_gpu.npz')['actions'][:460];rng=np.random.default_rng(144);best=-999
  with rollout.Rollout(nthread=8) as rr:
   for rd in range(5):
    A=np.tile(seed,(32,1,1));t=np.clip(np.arange(460)/100,0,1)
    A[:,:,:3]+=rng.uniform([-.003,-.004,-.003],[.003,.003,.003],(32,1,3))*t[None,:,None]
    A[:,:,10]+=rng.uniform(-.3,.3,(32,1))*t;A[:,:,14]+=rng.uniform(-.3,.3,(32,1))*t
    S,_=rr.rollout(mm,[mj.MjData(mm) for _ in range(8)],np.tile(s0,(32,1)),A)
    pos=S[:,:,1+j:4+j];score=pos[:,-60:,2].mean(1)-.4*np.linalg.norm(pos[:,-1,:2]-[-.093,.003],axis=1);i=score.argmax()
    if score[i]>best:
     best=score[i];np.save(out/'pick11.npy',A[i]);print(rd,best,pos[i,-1],flush=True)
  return
 traj=np.load(OUT/'positive_review/pebble_candidate_045_gpu.npz')['actions']
 traj=np.load(out/'guide_snapshot.npy')
 vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);cam=mj.MjvCamera();cam.lookat[:]=[-.098,.015,.036];cam.distance=.17;cam.elevation=-52;cam.azimuth=95
 rec=[];pairs=[];frames=[];streak=0;best=0;lifted=False;turned=False
 palm=[k.split('/')[-1] for k in m.body_label].index('link_0_0');mount=R.from_euler('x',90,degrees=True);bias=np.zeros(3)
 for t in range(len(traj)):
  cmd=base.copy()
  if True:cmd[active]=traj[min(t,len(traj)-1)]
  if mode=='place' and t>=455:
   x=s.body_q.numpy();o=x[ix];orient=R.from_quat(o[3:]);a=traj[454].copy()
   if t==455:start=o[:3].copy();initialrot=orient;lever=start-x[palm,:3]
   smooth=lambda z: (lambda u:u*u*(3-2*u))(np.clip(z,0,1))
   u=smooth((t-455)/130);down=smooth((t-600)/110);rel=smooth((t-715)/45);ret=smooth((t-760)/70)
   dest=start+u*(np.r_[xy,.061]-start);dest[2]-=.032*down
   delta=R.from_rotvec(u*(R.from_euler('z',yaw)*initialrot.inv()).as_rotvec())
   if t<715:bias=np.clip(bias+.055*(dest-o[:3]),-.012,.012)
   a[3:6]=(mount.inv()*delta*mount*R.from_euler('XYZ',a[3:6])).as_euler('XYZ')
   a[:3]+=mount.inv().apply(dest-start+lever-delta.apply(lever)+bias+[0,0,.045*ret])
   a[6:]+=(traj[170,6:]-a[6:])*rel;cmd[active]=a
  ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,1/240);s,n=n,s
  pose=s.body_q.numpy()[ix];rot=R.from_quat(pose[3:]);ang=np.rad2deg((R.from_euler('z',yaw).inv()*rot).magnitude())
  pts=rot.apply(np.c_[outline,np.full(5,-.004)])+pose[:3];depth=(.029-max(pts[:,2]))*1000
  contacts=set()
  for co in v.mj_data.contact:
   gs=[v.mj_model.geom(g).name for g in co.geom]
   if co.dist<=0 and any('wood_' in k for k in gs):pairs.append([t,*gs])
   if co.dist<=0 and any('/item/' in k for k in gs):contacts.update(k.split('/')[-2] for k in gs if '/item/' not in k)
  hold={'link_1_3','link_2_3'}<=contacts;touch=any(k.startswith('link_') for k in contacts)
  lifted |= hold and pose[2]>.0414
  turned |= lifted and hold and ang<=10
  seat=ang<=10 and 4.8<=depth<=6.5 and np.linalg.norm(pose[:2]-xy)<.002 and not touch
  streak=streak+1 if seat else 0;best=max(best,streak)
  rec.append([*pose.tolist(),ang,depth,hold,touch,seat])
  if t%8==0:
   q=s.joint_q.numpy()
   for j,k in enumerate(names):
    id=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,k)
    if id<0:continue
    d=vm.jnt_qposadr[id];h=b.joint_q_start[j]
    if vm.jnt_type[id]==0:vd.qpos[d:d+7]=np.r_[q[h:h+3],q[h+6],q[h+3:h+6]]
    else:vd.qpos[d]=q[h]
   mj.mj_forward(vm,vd);ren.update_scene(vd,cam);frames.append(ren.render().copy())
 subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r','30','-i','-','-pix_fmt','yuv420p',str(out/(mode+'.mp4'))],input=np.asarray(frames).tobytes(),check=True)
 (out/(mode+'_contacts.json')).write_text(json.dumps(pairs));np.save(out/(mode+'_actions.npy'),traj)
 result=dict(mode=mode,angle_deg=ang,depth_mm=depth,seated_s=best/240,pinched_lift=bool(lifted),held_aligned=bool(turned),success=bool(turned and streak>=120));(out/(mode+'.json')).write_text(json.dumps(result));np.save(out/(mode+'.npy'),rec);print(result);ren.close()
if __name__=='__main__':run(sys.argv[1])
