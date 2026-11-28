
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import OUT,np,E,mj
from surface_objects import geom
from scipy.spatial.transform import Rotation as R
import newton,json,sys,os,subprocess
from pathlib import Path
out=OUT/'stack09';out.mkdir(exist_ok=True)
xy=np.array([-.102,.034]);bp=np.r_[xy,.0131];up=np.array([-.093,.003,.0314])
A=np.array([[-10,-6],[4,-6],[10,-4],[10,1],[5,6],[-8,6],[-10,3.]])
B=np.array([[-20,-7],[-13,-14],[9,-14],[20,-5],[18,8],[7,14],[-12,12],[-20,4.]])
def scene():
 r=_evo_parse_xml(OUT/'pebble.xml').getroot();w=r.find('worldbody');asset=r.find('asset');hand=w.find('body')
 for b in list(w):
  if b.get('name','').startswith(('item','neighbor','rim','tray')):w.remove(b)
 r.remove(r.find('contact'));c=E.SubElement(r,'contact');geom(w,'pickup_pad','box',(-.093,.003,.024),'.017 .014 .003','.28 .34 .35 1')
 def stone(name,pos,yaw,verts,mass,col):
  from scipy.spatial import ConvexHull
  mass=2500*ConvexHull(np.vstack(verts)*.001).volume
  ob=E.SubElement(w,'body',name=name,pos=' '.join(map(str,pos)),euler=f'0 0 {yaw}')
  E.SubElement(ob,'freejoint',name=name+'_free');E.SubElement(asset,'mesh',name=name+'_hull',vertex=' '.join(map(str,np.vstack(verts).ravel()*.001)))
  geom(ob,name+'_rock','mesh',(0,0,0),'.01',col,mesh=name+'_hull',mass=str(mass));return ob
 def ring(a,z):return np.c_[a,np.broadcast_to(z,len(a))]
 a=stone('upper',up,.2,[ring(A*.87,-4),ring(A,[-2,-2.4,-2,-1,-1.8,-2,-2.5]),ring(A*.96,2.7),ring(A*.72+[.4,-.4],[4,4.4,4.6,4.8,4.5,4,3.9])],.0021,'.66 .43 .25 1')
 b=stone('lower',bp,.15,[ring(B*.86,-13),ring(B,-7),ring(B*.96,3),ring(B*.77+[.4,-.6],10)],.023,'.35 .40 .43 1')
 # Thin coplanar inner patches regularize flat-face contact; do not extend either hull.
 for ob,id,z,hx,hy,col in [(a,'upper_sole',-.0036,.005,.003,'.66 .43 .25 1'),(b,'lower_foot',-.0126,.010,.006,'.35 .40 .43 1'),(b,'lower_table',.0096,.009,.005,'.35 .40 .43 1')]:
  geom(ob,id,'box',(0,0,z),f'{hx} {hy} .0004',col,mass='.00001')
 pairs=set()
 for ob in (a,b):
  for g in ob.iter('geom'):
   for h in list(hand.iter('geom'))+w.findall('geom')+list((b if ob is a else a).iter('geom')):
    pairs.add(tuple(sorted([g.get('name'),h.get('name')])))
 for g,h in sorted(pairs):E.SubElement(c,'pair',geom1=g,geom2=h,friction='.9 .9 .003 .0001 .0001',solref='.009 1')
 p=out/'stack09.xml';E.ElementTree(r).write(p);return p
def run(mode):
 p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
 names=[k.split('/')[-1] for k in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};active=[]
 for i,k in enumerate(names):
  j=b.joint_qd_start[i]
  if 'diagnostic_wrist' in k:b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
  if k in valid and k.startswith(('wrist_','diagnostic_wrist_','link_')):active.append(j)
 if mode in ('face','edge','miss'):
  k=names.index('upper_free');j=b.joint_q_start[k];b.joint_q[j:j+7]=[*((np.r_[xy,.031]+([.034,.004,0] if mode=='miss' else [0,0,.007] if mode=='edge' else [0,0,0])).tolist()),*R.from_euler('x',90 if mode=='edge' else 0,degrees=True).as_quat().tolist()]
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40)
 bodies=[k.split('/')[-1] for k in m.body_label];ii,jj=map(bodies.index,['upper','lower'])
 traj=np.load(OUT/'inlay11/seated_release_actions.npy');traj=np.concatenate([traj,np.tile(traj[-1],(240,1))]);traj=traj if mode not in ('idle','face','edge','miss') else np.tile(base[active],(480,1))
 if mode=='custom':traj=np.load(out/'custom_actions.npy')
 vid='--video' in sys.argv;vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480) if vid else None;cam=mj.MjvCamera();cam.lookat[:]=[-.097,.019,.036];cam.distance=.17;cam.azimuth=75;cam.elevation=-30
 rows=[];frames=[];streak=0;lift=False;worst=0.;last=None;hist=[]
 for t,act in enumerate(traj):
  cmd=base.copy();cmd[active]=act;ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,1/240);s,n=n,s
  u,d=s.body_q.numpy()[[ii,jj]].astype(float);rel=R.from_quat(d[3:]).inv();pos=rel.apply(u[:3]-d[:3]);rot=rel*R.from_quat(u[3:]);tilt=np.rad2deg(np.arccos(np.clip(rot.as_matrix()[2,2],-1,1)))
  touch=held=on=external=False;contacts=set();hits=[]
  for co in v.mj_data.contact:
   if co.dist>0:continue
   g=[v.mj_model.geom(k).name for k in co.geom];bo=[k.split('/')[-2] for k in g]
   if not any(k in ('upper','lower') for k in bo):continue
   hits.append(g)
   if any(k.startswith('link_') for k in bo):
    touch=True
    if 'upper' in bo:contacts.update(bo)
   if set(bo)=={'upper','lower'}:on=True
   if 'upper' in bo and 'worldbody' in bo:external=True
  held={'link_1_3','link_2_3'}<=contacts;lift|=held and u[2]-up[2]>.010
  motion=1000*np.linalg.norm(d[:3]-bp);worst=max(worst,motion)
  # Relative pose of the intended bearing faces; reject edge/side balances.
  eligible=on and not touch and not external and tilt<=10 and abs(pos[2]-.014)<.0015 and np.linalg.norm(pos[:2])<.011
  streak=streak+1 if eligible else 0
  rows.append([*u,*d,motion,tilt,int(touch),int(held),int(on),int(eligible)]);hist.append(hits)
  if ren and t%8==0:
   q=s.joint_q.numpy()
   for k,name in enumerate(names):
    j=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)
    if j<0:continue
    dst=vm.jnt_qposadr[j];src=b.joint_q_start[k]
    if vm.jnt_type[j]==0:vd.qpos[dst:dst+7]=np.r_[q[src:src+3],q[src+6],q[src+3:src+6]]
    else:vd.qpos[dst]=q[src]
   mj.mj_forward(vm,vd);ren.update_scene(vd,cam);left=ren.render().copy()
   near=mj.MjvCamera();near.lookat[:]=[-.099,.022,.032];near.distance=.115;near.azimuth=155;near.elevation=-18
   hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];vm.geom_group[hs]=5;opt=mj.MjvOption();opt.geomgroup[5]=0;ren.update_scene(vd,near,scene_option=opt);right=ren.render().copy();vm.geom_group[hs]=0
   from PIL import Image,ImageDraw
   im=Image.fromarray(np.hstack([left,right]));draw=ImageDraw.Draw(im);draw.rectangle([0,0,960,22],fill=(24,34,38));draw.text((8,4),f'hand_111 | Newton | {t/240:.2f} s',fill='white');draw.text((487,4),f'Stones only | base max {worst:.2f} mm | released + stable {streak/240:.2f} s',fill='white');frames.append(np.array(im))
 rows=np.array(rows);tail=rows[-240:];wander=1000*np.linalg.norm(tail[:,:3]-tail[0,:3],axis=1).max()
 res=dict(lift=bool(lift),max_lift_mm=float(1000*(rows[rows[:,17]>0,2].max()-up[2])) if rows[:,17].any() else 0.,lower_max_mm=worst,stable_s=streak/240,tilt_deg=tilt,final_upper_motion_mm=wander,success=bool(lift and worst<=10 and streak>=240 and wander<=1))
 np.save(out/(mode+'.npy'),rows);(out/(mode+'.json')).write_text(json.dumps(res));(out/(mode+'_contacts.json')).write_text(json.dumps(hist));print(res,flush=True)
 if ren:
  subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(out/(mode+'.mp4'))],input=np.asarray(frames).tobytes(),check=True);ren.close()
if __name__=='__main__':run(sys.argv[1])
