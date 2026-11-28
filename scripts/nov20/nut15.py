
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
from scipy.optimize import least_squares
import newton,sys,os,json
out=OUT/'nut15';out.mkdir(exist_ok=True)
pitch=.00175;cx=-.080;cy=-.009;z=.077;h=.005;rr=.0172;hole=.0064
def scene():
 r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');hand=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'))
 for k in ['wrist_roll','wrist_pitch','wrist_yaw']:r.find(f".//joint[@name='{k}']").set('range','-.04363323 .04363323')
 geom(w,'base','box',(cx,cy,.033),'.023 .023 .033','.25 .38 .40 1')
 top=z-h+pitch;bottom=.066
 geom(w,'stud','cylinder',(cx,cy,(bottom+top)/2),f'.006 {(top-bottom)/2}', '.36 .39 .40 1')
 # Inner hole polygon (12 sectors); hex vertices alternate with midpoints.
 root=E.SubElement(w,'body',name='guide',pos=f'{cx} {cy} {z}');E.SubElement(root,'joint',name='feed',type='slide',axis='0 0 -1',damping='.03',range='-.00175 .006',armature='.00001')
 E.SubElement(root,'inertial',pos='0 0 0',mass='.0001',diaginertia='1e-8 1e-8 1e-8')
 nut=E.SubElement(root,'body',name='nut');E.SubElement(nut,'joint',name='turn',type='hinge',axis='0 0 -1',limited='false',damping='.00014',frictionloss='.0016',solimpfriction='.999 .9999 .0001 .5 2',solreffriction='.003 1',armature='.000018')
 for i in range(12):
  pts=[]
  for k in [i,i+1]:
   ang=k*np.pi/6;v=np.array([np.cos(ang),np.sin(ang)]);rad=rr*(np.cos(np.pi/6) if k%2 else 1)
   for zz,ro in [(-h,rad-.0007),(-h+.0008,rad),(h-.0008,rad),(h,rad-.0007)]:pts.append([*(v*ro),zz])
   for zz in [-h,h]:pts.append([*(v*hole),zz])
  name=f'sector{i}';E.SubElement(r.find('asset'),'mesh',name=name,vertex=' '.join(map(str,np.ravel(pts))))
  geom(nut,name,'mesh',(0,0,0),'.01','.84 .53 .17 1' if i in (0,11) else '.52 .58 .60 1',mesh=name,mass='.0034')
 eq=E.SubElement(r,'equality');E.SubElement(eq,'joint',name='screw_lead',joint1='turn',joint2='feed',polycoef=f'0 {2*np.pi/pitch} 0 0 0',solref='.002 1',solimp='.995 .999 .0001 .5 2')
 con=E.SubElement(r,'contact')
 for g in hand.iter('geom'):
  for n in [f'sector{i}' for i in range(12)]+['stud','base']:E.SubElement(con,'pair',geom1=g.get('name'),geom2=n,friction='.85 .85 .001 .0001 .0001',solref='.007 1')
 p=out/'nut15.xml';E.ElementTree(r).write(p);return p
def run(mode):
 p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);names=[k.split('/')[-1] for k in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};ids=[]
 for i,k in enumerate(names):
  j=b.joint_qd_start[i]
  if k in ('turn','feed'):b.joint_target_ke[j]=b.joint_target_kd[j]=0
  elif k in valid:
   ids.append(j)
   if 'diagnostic_wrist' in k:b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
   elif k in ('wrist_roll','wrist_pitch','wrist_yaw'):b.joint_target_ke[j]=9.;b.joint_target_kd[j]=.20
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=60);mm=v.mj_model;dd=v.mj_data;print('EQUALITY',mm.neq,mm.eq_data,flush=True)
 if mm.neq!=1:raise RuntimeError('missing lead coupling')
 gn=[mm.geom(i).name for i in range(mm.ngeom)];ix=[k.split('/')[-1] for k in m.body_label].index('nut');js=[b.joint_q_start[names.index(k)] for k in ['turn','feed','wrist_roll','wrist_pitch','wrist_yaw']]
 vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);hz=480;stroke=.64;approach=.60;N=int((approach+20*stroke+.5)*hz) if mode=='hand' else 720;rows=[];contact=[];faces=[];qs=[];actions=[]
 # IK tables: alternate two pads in clockwise arcs; wrist remains at neutral.
 stop=None
 if mode=='hand':
  from scipy.interpolate import RegularGridInterpolator
  tables=[]; xs=np.linspace(-1,1,19);zs=np.linspace(-.008,.020,13)
  for finger in (0,1):
   joints=np.arange(11,15)+4*finger;g=vm.geom('handgeom'+str(40+20*finger)).id;lim=vm.jnt_range[joints];grid=np.zeros((len(xs),len(zs),4))
   for ii,u in enumerate(xs):
    for jj,zz in enumerate(zs):
     sign=1 if finger==0 else -1;ang=u*.78;target=np.array([cx+sign*.012*np.sin(ang),cy+sign*.012*np.cos(ang),z+h+.005+zz])
     def fun(q):
      vd.qpos[:27]=0;vd.qpos[joints]=q;mj.mj_forward(vm,vd)
      return np.r_[(vd.geom_xpos[g]-target)*1000,.15*(q[2]-q[3])]
     last=grid[ii,jj-1] if jj else grid[ii-1,jj] if ii else np.array([0,.15,.6,.8])
     res=least_squares(fun,np.clip(last,*lim.T),bounds=lim.T,max_nfev=45);grid[ii,jj]=res.x
   tables.append(RegularGridInterpolator((xs,zs),grid,bounds_error=False,fill_value=None))
  print('IK done',flush=True)
 for t in range(N):
  sec=t/hz;cmd=base.copy();s.clear_forces()
  if mode in ('push','torque','subtorque'):
   force=np.zeros((m.body_count,6),np.float32);load=min(1,sec/.15)
   if mode=='push':force[ix,2]=-3*load
   else:force[ix,5]=-(.003 if mode=='torque' else .0005)*load
   s.body_f.assign(force)
  if mode=='hand':
   sm=lambda a:(lambda u:u*u*(3-2*u))(np.clip(a,0,1));timer=np.clip((sec-approach)/stroke,0,19.99999);no=int(timer);phase=timer-no;act=np.zeros(27)
   depth=s.joint_q.numpy()[js[1]];press=float(os.getenv('PRESS','-.002'))
   for finger,interp in enumerate(tables):
    if no%2==finger: u=-1+2*sm((phase-.10)/.75);zz=.012+(press-.012)*sm((phase+.1)/.20)+(.012-press)*sm((phase-.85)/.15)
    else:u=1-2*sm(phase/.78);zz=.012
    qq=interp([[u,np.clip(zz-depth,-.008,.020)]])[0];act[11+finger*4:15+finger*4]=qq*sm(sec/approach)
   if stop is None and s.joint_q.numpy()[js[0]]>=2.13*2*np.pi:stop=sec;release=act.copy()
   if stop is not None:
    act=release.copy();act[1]=.031*sm((sec-stop)/.35)
   cmd[ids]=act
  ctrl.joint_target_q.assign(cmd);v.step(s,n,ctrl,None,1/hz);s,n=n,s;q=s.joint_q.numpy();F=np.zeros(5);tau=np.zeros(5);cnt=np.zeros(5);fac=np.zeros(5,dtype=int)
  for j,co in enumerate(dd.contact):
   ge=[gn[k] for k in co.geom];bo=[k.split('/')[-2] for k in ge]
   if 'nut' not in bo or co.dist>0 or co.efc_address<0:continue
   k=bo.index('nut');oth=bo[1-k]
   if not oth.startswith('link_') or not oth[5:6].isdigit() or oth[5]=='0':continue
   finger=int(oth[5])-1;f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
   if f[0]<.001:continue
   world=co.frame.reshape(3,3).T@f[:3]*(1 if k==1 else -1);fac[finger]|=1<<int(ge[k].split('/')[-1].split('_')[0][6:]);F[finger]+=f[0];tau[finger]-=np.cross(co.pos-np.array([cx,cy,z]),world)[2];cnt[finger]+=1
  rows.append([sec,q[js[0]],z-s.body_q.numpy()[ix,2],q[js[1]],*q[js[2:]],*F,*tau,*cnt]);actions.append(cmd[ids].copy());faces.append(fac)
  if t%2==0:qs.append(q.copy())
  if mode=='hand' and stop is not None and sec-stop>.70:break
 x=np.array(rows);np.savez_compressed(out/(mode+'.npz'),data=x,q=qs,names=names,starts=b.joint_q_start,actions=actions,faces=faces);print(mode,x[-1,:7],x[:,7:12].max(0), 'lead_err',np.max(abs(x[:,2]-pitch*x[:,1]/(2*np.pi))),flush=True)
if __name__=='__main__':run(sys.argv[1])
