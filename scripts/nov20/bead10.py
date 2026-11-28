
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
from scipy.spatial import ConvexHull
out=OUT/'bead10';out.mkdir(exist_ok=True)
xy=np.array([-.102,.034]);up=np.array([-.093,.003,.0332]);zt=.038;zb=.021;H=.006;rad=.0016;hole=.003;N=12
def scene():
 r=_evo_parse_xml(OUT/'pebble.xml').getroot();w=r.find('worldbody');asset=r.find('asset');hand=w.find('body')
 for b in list(w):
  if b.get('name','').startswith(('item','neighbor','rim','tray')):w.remove(b)
 r.remove(r.find('contact'));c=E.SubElement(r,'contact')
 geom(w,'pickup_pad','box',(-.093,.003,.024),'.017 .014 .003','.28 .34 .35 1')
 geom(w,'plinth','box',(*xy,zb/2),'.015 .013 .0105','.27 .39 .44 1')
 geom(w,'rod','cylinder',(*xy,(zt+zb)/2),f'{rad} {(zt-zb)/2}', '.72 .41 .20 1')
 ob=E.SubElement(w,'body',name='bead',pos=' '.join(map(str,up)),euler='0 0 .2');E.SubElement(ob,'freejoint',name='bead_free')
 for i in range(N):
  pts=[]
  for k in [i,i+1]:
   a=2*np.pi*k/N;v=np.array([np.cos(a),np.sin(a)])
   for z,scale in [(-H,.83),(-.002,1),(.002,1),(H,.83)]:pts.append([*(v*np.array([.010,.0072])*scale),z])
   for z in [-H,H]:pts.append([*(v*hole),z])
  vs=np.array(pts);name='arc'+str(i);E.SubElement(asset,'mesh',name=name,vertex=' '.join(map(str,vs.ravel())))
  geom(ob,name,'mesh',(0,0,0),'.01',f'{.77+.018*np.cos(i):.3f} .66 .46 1',mesh=name,mass=str(1900*ConvexHull(vs).volume))
  for g in list(hand.iter('geom'))+w.findall('geom'):
   E.SubElement(c,'pair',geom1=name,geom2=g.get('name'),solref='.006 1',friction='.85 .85 .002 .0001 .0001')
 for g in hand.iter('geom'):
  for dest in ('rod','plinth'):E.SubElement(c,'pair',geom1=g.get('name'),geom2=dest,solref='.007 1')
 p=out/'bead10.xml';E.ElementTree(r).write(p);return p
def measure(pose):
 # Support-function bound for FULL rod cross section, including inclined projection,
 # over the axial overlap; polygon face normals are expressed in bead coordinates.
 rot=R.from_quat(pose[3:]).inv();d=rot.apply([0,0,1]);tip=rot.apply(np.r_[xy,zt]-pose[:3]);foot=tip-(zt-zb)*d
 if d[2]<.15:return 0.,-1.,180.
 lo=max(-H,foot[2]-rad*np.linalg.norm(d[:2]));hi=min(H,tip[2]+rad*np.linalg.norm(d[:2]))
 tilt=np.rad2deg(np.arccos(np.clip(d[2],-1,1)))
 if hi<=lo:return 0.,1.,tilt
 normals=np.c_[np.cos((np.arange(N)+.5)*2*np.pi/N),np.sin((np.arange(N)+.5)*2*np.pi/N)]
 sl=d[:2]/d[2];spread=rad*np.sqrt(1+(normals@sl)**2)
 gap=min(np.min(hole*np.cos(np.pi/N)-normals@(tip[:2]+sl*(z-tip[2]))-spread) for z in (lo,hi))
 return np.clip(tip[2]+H,0,2*H)*1000,gap*1000,tilt
def run(mode):
 p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
 names=[k.split('/')[-1] for k in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};active=[]
 for i,k in enumerate(names):
  j=b.joint_qd_start[i]
  if 'diagnostic_wrist' in k:b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
  if k in valid and k.startswith(('wrist_','diagnostic_wrist_','link_')):active.append(j)
 fine=mode.startswith(('guide','free'));fixture=mode.startswith('guide');probe=fine or mode=='aligned' or mode.startswith('skew')
 if probe:
  j=b.joint_q_start[names.index('bead_free')];ang=0 if mode=='aligned' else float(mode[5:] if fixture else mode[4:] or 25);rr=R.from_euler('y',ang,degrees=True);center=np.r_[xy,zt+.0002]-rr.apply([0,0,-H]);b.joint_q[j:j+7]=[*center,*rr.as_quat()]
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50)
 idx=[k.split('/')[-1] for k in m.body_label].index('bead');traj=np.tile(base[active],(1440 if mode.startswith('skew') else 1440 if fine else 360,1)) if probe else np.load(out/('actions.npy' if mode=='hand' else mode+'_actions.npy'))
 vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480) if '--video' in sys.argv else None;cam=mj.MjvCamera();cam.lookat[:]=[-.099,.018,.041];cam.distance=.165;cam.azimuth=75;cam.elevation=-30
 rows=[];frames=[];qs=[];hist=[]
 for t,act in enumerate(traj):
  cmd=base.copy();cmd[active]=act;ctrl.joint_target_q.assign(cmd)
  for sub in range(4 if fine else 1):
   s.clear_forces()
   if fixture:
    pose0=s.body_q.numpy()[idx];err=(rr*R.from_quat(pose0[3:]).inv()).as_rotvec()
    # External orientation-only torsional test fixture. No translational forces,
    # pose targets or bead joint drives. Small-angle relative rotation for damping.
    vel=np.zeros(3) if t==0 and sub==0 else (R.from_quat(pose0[3:])*prev.inv()).as_rotvec()*960
    prev=R.from_quat(pose0[3:]);tau=np.clip(.009*err-.000025*vel,-.00065,.00065);f=np.zeros((m.body_count,6),np.float32);f[idx,3:]=tau;s.body_f.assign(f)
   v.step(s,n,ctrl,None,1/(960 if fine else 240));s,n=n,s
  pose=s.body_q.numpy()[idx];dep,gap,tilt=measure(pose);contacts=set();pen=0.;arcs=set()
  for co in v.mj_data.contact:
   if co.dist>0:continue
   g=[v.mj_model.geom(k).name for k in co.geom];bo=[k.split('/')[-2] for k in g]
   if 'bead' not in bo:continue
   other=g[1-bo.index('bead')];contacts.add(other.split('/')[-2]);contacts.add(other.split('/')[-1].rsplit('_',1)[0])
   if '/rod_' in other:pen=max(pen,-co.dist*1000);arcs.add(g[bo.index('bead')].split('/')[-1].rsplit('_',1)[0])
  held={'link_1_3','link_2_3'}<=contacts;hand=any(k.startswith('link_') for k in contacts)
  rows.append([*pose,dep,gap,tilt,held,hand,pen,int('rod' in contacts),int('plinth' in contacts)]);hist.append(sorted(arcs));qs.append(s.joint_q.numpy().copy())
  if ren and t%8==0:
   q=qs[-1]
   for k,name in enumerate(names):
    j=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)
    if j<0:continue
    dst=vm.jnt_qposadr[j];src=b.joint_q_start[k]
    if vm.jnt_type[j]==0:vd.qpos[dst:dst+7]=np.r_[q[src:src+3],q[src+6],q[src+3:src+6]]
    else:vd.qpos[dst]=q[src]
   mj.mj_forward(vm,vd);ren.update_scene(vd,cam);left=ren.render().copy()
   near=mj.MjvCamera();near.lookat[:]=[*xy,.035];near.distance=.088;near.azimuth=145;near.elevation=-42
   hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];vm.geom_group[hs]=5;opt=mj.MjvOption();opt.geomgroup[5]=0;ren.update_scene(vd,near,scene_option=opt);right=ren.render().copy();vm.geom_group[hs]=0
   from PIL import Image,ImageDraw
   im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,22],fill=(24,34,38));dr.text((8,4),f'{mode} | Newton | {t/240:.2f} s',fill='white');dr.text((488,4),f'Bore close-up | tilt {tilt:.1f} deg',fill='white');frames.append(np.array(im))
 x=np.array(rows);np.save(out/(mode+'.npy'),x);(out/(mode+'_contacts.json')).write_text(json.dumps(hist));print(mode,x[-1], 'max pen',x[:,12].max(),'max lift',(x[x[:,10]>0,2].max()-up[2])*1000 if x[:,10].any() else 0,flush=True)
 if ren:
  subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(out/(mode+'.mp4'))],input=np.array(frames).tobytes(),check=True);ren.close()
if __name__=='__main__':run(sys.argv[1])
