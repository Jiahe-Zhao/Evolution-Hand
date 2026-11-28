
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import OUT,np,E,mj
import newton,sys,os
from scipy.spatial.transform import Rotation
o=OUT/'pen14';o.mkdir(exist_ok=True)
D=np.load(OUT/'pen14/warm.npz');q=D['q'][:34].copy();pos=q[27:30].copy();rot=Rotation.from_quat(q[30:34]);fmt=lambda v:' '.join(map(str,v))
mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;r=_evo_parse_xml(OUT/'stack09/stack09.xml').getroot();w=r.find('worldbody');h=w.find('body');ob=w.find("body[@name='upper']");ob.set('name','pen')
w.remove(w.find("body[@name='lower']"))
for g in list(w.findall('geom')):
 if g.get('name')!='floor':w.remove(g)
r.remove(r.find('contact'));c=E.SubElement(r,'contact');ob.find('geom').set('name','grip');ob.find('geom').set('rgba','.14 .38 .57 1')
# Offset nib towards the free side of the grip.
pt=lambda xyz:rot.inv().apply(xyz)
def geom(b,name,typ,**kw):return E.SubElement(b,'geom',name=name,type=typ,**kw)
geom(ob,'barrel','capsule',fromto=fmt(np.r_[pt([0,0,-.001]),pt([-.001,0,-.024])]),size='.0038',rgba='.15 .36 .57 1',mass='.002')
geom(ob,'nib','capsule',fromto=fmt(np.r_[pt([-.002,0,-.030]),pt([-.001,0,-.024])]),size='.0013',rgba='.57 .55 .50 1',mass='.0004')
geom(ob,'tip','sphere',pos=fmt(pt([-.002,0,-.0314])),size='.001',rgba='.09 .10 .12 1',mass='.0001')
geom(w,'paper','box',pos=fmt([pos[0]+.017,pos[1],pos[2]-.0354-(.010 if mode=='hover' else 0)]),size='.085 .060 .001',rgba='.94 .93 .88 1')
def pair(a,b,mu):E.SubElement(c,'pair',geom1=a,geom2=b,friction=f'{mu} {mu} .001 .0001 .0001',solref=(os.getenv('SOFT','.04 1.4') if {a,b}=={'tip','paper'} else '.005 1'))
for g in ob.findall('geom'):
 pair(g.get('name'),'paper',.12)
 for hg in h.iter('geom'):pair(g.get('name'),hg.get('name'),.95)
for hg in h.iter('geom'):pair(hg.get('name'),'paper',.5)

p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);b.joint_q[:]=list(q)
for j,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k.split('/')[-1]:i=b.joint_qd_start[j];b.joint_target_ke[i]=1700;b.joint_target_kd[i]=42
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=45);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=D['act'];mm=v.mj_model;dd=v.mj_data;names=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];tip=names.index('tip');dt=1/960;xy=np.array([-.091,.003]);off=np.zeros(3);rows=[];qs=[]
for t in range(3700 if mode=='hover' else 6300):
 u=np.clip((t-600)/3000 if mode=='hover' else (t-2160)/3960,0,1);target=np.array([pos[0]-.002+.046*u,pos[1]+.006*np.sin(2*np.pi*u)]);cmd=base.copy();cmd[:3]+=off;cmd[0]+=.046*u*(mode!='hover');cmd[2]-=.006*np.sin(2*np.pi*u)*(mode!='hover');ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,dt);s,n=n,s;xyz=dd.geom_xpos[tip];fn=0;gr=[0.,0.];contact=np.array([np.nan,np.nan])
 for j,co in enumerate(dd.contact):
  if co.dist>0 or co.efc_address<0:continue
  i,k=co.geom;gg={names[i],names[k]};f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
  if gg=={'tip','paper'} and f[0]>0:fn+=f[0];contact=co.pos[:2].copy()
  if gg.intersection({'grip','gripflare0','gripflare1','barrel'}):
   for z in (i,k):
    part=mm.geom(z).name.split('/')[-2];
    if part.startswith(('link_1_','link_2_')):gr[int(part[5])-1]+=f[0]
 off[0]+=.006*(target[0]-xyz[0]);off[2]-=.006*(target[1]-xyz[1])
 if mode=='hover':off[1]+=.0125*(pos[2]-.0314-xyz[2])
 else:off[1]+=np.clip(float(os.getenv('GAIN','.00008'))*(fn-float(os.getenv('FORCE','.035'))),-.00008,.00008)
 off=np.clip(off,[-.025,-.025,-.028],[.069,.025,.028]);rows.append([t*dt,u,*target,*xyz,*contact,fn,*gr]);qs.append(s.joint_q.numpy().copy())
np.save(o/'origin.npy',pos);a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs);print(mode,'stride',a[::240,[1,4,5,6,9,10,11]],flush=True)
