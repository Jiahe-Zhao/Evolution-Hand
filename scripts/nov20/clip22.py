
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import newton,sys,os,csv
o=OUT/'clip22';o.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;fmt=lambda x:' '.join(map(str,x));ref=OUT/'pen14/write.xml'
r=_evo_parse_xml(ref).getroot();w=r.find('worldbody');hand=w.find('body')
for b in list(w):
 if b.tag not in ('light','camera') and b is not hand and b.get('name')!='floor':w.remove(b)
r.remove(r.find('contact'));c=E.SubElement(r,'contact')
q0=np.load(OUT/'pen14/warm.npz')['q'][:27].copy();vm=mj.MjModel.from_xml_path(str(ref));vd=mj.MjData(vm);vd.qpos[:27]=q0;mj.mj_forward(vm,vd);O=(vd.geom('handgeom20').xpos+vd.geom('handgeom40').xpos)*.5
plug=E.SubElement(w,'body',name='plug',pos=fmt(O));E.SubElement(plug,'freejoint',name='plugfree');fixed=[]
def box(b,n,xyz,size,col,mass=None):
 pos=np.array(xyz)+(O if b is w else 0);g=E.SubElement(b,'geom',name=n,type='box',pos=fmt(pos),size=fmt(size),rgba=col,friction='.85 .003 .0002',contype='2',conaffinity='3');
 if mass:g.set('mass',str(mass))
 if b is w:fixed.append(n)
 return g
blue='.15 .45 .61 1';grey='.32 .42 .45 1';colors=['.93 .51 .14 1','.22 .72 .53 1']
box(plug,'spineLower',[0,0,-.025],[.006,.0013,.017],blue,.004)
box(plug,'spineUpper',[0,0,.002],[.006,.0013,.006],blue,.002)
for sg in (-1,1):box(plug,'rail'+str(sg),[sg*.0048,0,-.006],[.0012,.0013,.002],blue,.0005)
box(w,'landing',[.040,0,-.025],[.021,.020,.002],'.24 .36 .39 1')
box(w,'leg',[.040,0,-.041],[.017,.016,.014],grey)
box(plug,'shank',[0,0,-.035],[.0068,.0081,.010],blue,.005)
box(w,'base',[0,0,-.047],[.012,.015,.002],grey)
for sg in (-1,1):
 box(w,'guideX'+str(sg),[sg*.0083,0,-.0305],[.0012,.012,.0145],grey)
 box(w,'guideY'+str(sg),[0,sg*.0098,-.035],[.0072,.0013,.010],grey)
 box(w,'wall'+str(sg),[0,sg*.013,-.0225],[.0072,.001,.0075],grey)
 box(w,'catch'+str(sg),[0,sg*.0107,-.017],[.0072,.0022,.0015],'.55 .58 .59 1')
for i,sg in enumerate([1,-1]):
 b=E.SubElement(plug,'body',name='ear'+str(i));E.SubElement(b,'joint',name='travel'+str(i),type='slide',axis=fmt([0,-sg,0]),range='0 .0038',damping='.65',armature='.0008')
 box(b,'runner'+str(i),[sg*.002,sg*.0034,-.006],[.0013,.0034,.0013],colors[i],.0002)
 box(b,'arm'+str(i),[0,sg*.0066,-.010],[.004,.0008,.012],colors[i],.0008)
 box(b,'hook'+str(i),[0,sg*.00845,-.0206],[.004,.0021,.0015],colors[i],.0007)
 box(b,'pad'+str(i),[0,sg*.0089,.001],[.009,.0017,.0065],colors[i],.002)
 box(b,'lip'+str(i),[0,sg*.0099,.008],[.009,.0027,.001],colors[i],.0005)
def pair(x,y,mu='.8'):E.SubElement(c,'pair',geom1=x,geom2=y,friction=f'{mu} {mu} .001 .0001 .0001',solref='.004 1')
for hg in hand.iter('geom'):
 for g in plug.iter('geom'):pair(hg.get('name'),g.get('name'),'1')
for g in plug.iter('geom'):
 for n in fixed:pair(g.get('name'),n,'.32')
# Only the diagnostic fixture trials disable hand contact by moving the wrist away.
p=o/(tag+'.xml');E.ElementTree(r).write(p)
vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);vd.qpos[:27]=q0;tipids=[vm.geom('handgeom20').id,vm.geom('handgeom40').id]
def fit(compress):
 x0=q0[6:15].copy();ids=[[vm.geom(a+str(i)).id for a in ('pad','lip')] for i in range(2)];lim=vm.jnt_range[6:15]
 def loss(x):
  vd.qpos[6:15]=x;mj.mj_kinematics(vm,vd);z=np.zeros(6);out=[]
  for i,ti in enumerate(tipids):
   ds=mj.mj_geomDistance(vm,vd,ti,ids[i][0],.06,z);out.append(ds*1000+compress)
   out.extend(.12*(z[:3]-O)[[0,2]]*1000)
  return np.r_[out,.22*(x-x0)]
 return least_squares(loss,np.clip(x0,lim[:,0]+.00001,lim[:,1]-.00001),bounds=(lim[:,0],lim[:,1]),diff_step=.0001,max_nfev=80).x
rest=fit(-.8);push=fit(float(os.getenv('DEP','3.5')));op=fit(-5.5)
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);ns=[k.split('/')[-1] for k in b.joint_label];js=lambda k:b.joint_q_start[ns.index(k)];ks=lambda k:b.joint_qd_start[ns.index(k)];jc=js('plugfree');je=[js('travel'+str(i)) for i in range(2)];ke=[ks('travel'+str(i)) for i in range(2)]
for k in ke:b.joint_target_ke[k]=b.joint_target_kd[k]=0
for k in range(3):b.joint_target_ke[k]=2200;b.joint_target_kd[k]=46
q=np.array(b.joint_q);q[:27]=q0;q[6:15]=rest
if mode!='hand':q[0]+=.16
b.joint_q[:]=q.tolist();m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=60);ct=m.control();base=ct.joint_target_q.numpy().copy();base[:27]=q[:27];mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];bi=[k.split('/')[-1] for k in m.body_label].index('plug')
smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));dt=1/960;rows=[];qs=[];contacts=[]
for t in range(2400 if mode=='hand' else 1850):
 T=t*dt;cmd=base.copy();f=np.zeros(m.joint_dof_count,np.float32);q=s.joint_q.numpy();f[ke]=-float(os.getenv('K','90'))*q[je]-4000*(q[je]-np.clip(q[je],0,.0038));s.clear_forces()
 if mode=='hand':
  cmd[6:15]=rest+(push-rest)*smooth((T-.12)/.38)+(op-push)*smooth((T-1.8)/.32)
  cmd[1]+=.044*smooth((T-.62)/.7);cmd[0]+=.035*smooth((T-1.38)/.36)
 else:
  if mode in ('left','both'):f[ke[0]]+=.30*smooth(T/.24)*(1-smooth((T-1.48)/.14))
  if mode in ('right','both'):f[ke[1]]+=.30*smooth(T/.24)*(1-smooth((T-1.48)/.14))
  ext=np.zeros((m.body_count,6),np.float32);ext[bi,2]=.50*smooth((T-.4)/.18)*(1-smooth((T-1.28)/.14));s.body_f.assign(ext)
 ct.joint_f.assign(f);ct.joint_target_q.assign(cmd);v.step(s,n,ct,None,dt);s,n=n,s;q=s.joint_q.numpy().copy();ld=np.zeros((2,3));hook=np.zeros(2)
 for j in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[j];gs=[gn[i] for i in co.geom];F=np.zeros(6);mj.mj_contactForce(mm,dd,j,F)
  if F[0]<.001:continue
  if any(g.startswith('handgeom') for g in gs):
   jj=next(k for k,g in enumerate(gs) if g.startswith('handgeom'));bod=mm.geom(int(co.geom[1-jj])).name.split('/')[-2]
   if bod in ('plug','ear0','ear1') and not gs[1-jj].startswith(('pad','lip','arm')):contacts.append([T,-1,mm.geom(int(co.geom[jj])).name.split('/')[-2],gs[jj],gs[1-jj],F[0]])
  for i in range(2):
   if set(gs)=={'hook'+str(i),'catch'+str([1,-1][i])}:hook[i]+=F[0]
   for a in ('pad','lip','arm'):
    name=a+str(i)
    if name not in gs:continue
    k=1-gs.index(name);body=mm.geom(int(co.geom[k])).name.split('/')[-2]
    if gs[k].startswith('handgeom'):
     ld[i,0 if body=='link_1_3' else 1 if body=='link_2_3' else 2]+=F[0];contacts.append([T,i,body,gs[k],name,F[0]])
 rows.append([T,*((q[jc:jc+3]-O)*1000),*(q[je]*1000),*ld.ravel(),*hook]);qs.append(q)
 if mode=='both' and rows[-1][3]>=20:break
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs);np.savetxt(o/(tag+'_trace.csv'),a,delimiter=',',comments='',header='s,x_mm,y_mm,pull_mm,L_mm,R_mm,Lthumb_N,Lindex_N,Lother_N,Rthumb_N,Rindex_N,Rother_N,Lhook_N,Rhook_N')
with open(o/(tag+'_contacts.csv'),'w') as ff:
 cw=csv.writer(ff);cw.writerow(['time_s','ear','hand_link','hand_geom','ear_part','normal_N']);cw.writerows(contacts)
print(tag,'max',a[:,3:].max(0).round(3),'stages',a[::240,3:12].round(2),flush=True)
