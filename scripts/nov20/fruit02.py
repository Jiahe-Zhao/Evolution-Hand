
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
import newton,sys,csv,os
from scipy.spatial.transform import Rotation as R
out=OUT/'fruit02';out.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;C=np.array([-.093,.003,.083]);anchor=C+[0,0,.044];L=.030;tip=np.array([0,0,.014]);dt=1/480;threshold=.45
r=_evo_parse_xml(OUT/'pebble.xml').getroot();w=r.find('worldbody');hand=w.find('body')
for e in list(w):
 if (e.tag=='body' and e!=hand) or (e.tag=='geom' and e.get('name')!='floor'):w.remove(e)
if r.find('contact') is not None:r.remove(r.find('contact'))
c=E.SubElement(r,'contact')
E.SubElement(w,'geom',name='branch',type='capsule',fromto='-.195 .016 .135 -.093 .003 .127',size='.004',rgba='.37 .21 .10 1',contype='2',conaffinity='3')
bdy=E.SubElement(w,'body',name='fruit',pos='-.093 .003 .083');E.SubElement(bdy,'freejoint',name='fruitfree');E.SubElement(bdy,'geom',name='berry',type='ellipsoid',size='.015 .014 .014',mass='.012',rgba='.9 .16 .045 1',contype='2',conaffinity='3')
# Flatten the two cheeks of the oblate fruit; convex ring mesh follows the same envelope.
theta=np.linspace(.02,np.pi-.02,13);phi=np.arange(24)*np.pi/12
verts=np.array([[.015*np.sin(u)*np.cos(v),.014*np.clip(np.cos(u),-.86,.86),.014*np.sin(u)*np.sin(v)] for u in theta for v in phi]);E.SubElement(r.find('asset'),'mesh',name='fruitSkin',vertex=' '.join(map(str,verts.ravel())));bg=bdy.find('geom');bg.set('type','mesh');bg.set('mesh','fruitSkin');bg.attrib.pop('size')
for g in hand.iter('geom'):E.SubElement(c,'pair',geom1=g.get('name'),geom2='berry',friction='.85 .85 .002 .0001 .0001',condim='6',solref='.008 1')
p=out/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for j in range(3):b.joint_target_ke[j]=2500;b.joint_target_kd[j]=50
arr=np.load(OUT/('opposed_pebble_'+os.getenv('PH','1.571')+'.npz'))['arr'];q=np.array(b.joint_q);q[:15]=arr[0];q[0]+=float(os.getenv('X','0'));q[1]+=float(os.getenv('Z','.049'));b.joint_q[:]=list(q);m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=q[:27];v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];bi=next(i for i,x in enumerate(b.body_label) if x.endswith('/fruit'));palm=next(i for i,x in enumerate(b.body_label) if x.endswith('/link_0_0'));smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1));prev=L;attached=True;rows=[];contacts=[];qs=[]
for k in range(1200):
 t=k*dt;u=(float(os.getenv('CLOSE','13')) if mode=='pick' else float(os.getenv('TAP','5.5')))*smooth((t-.25)/.45)
 if mode!='pick':u*=1-smooth((t-.8)/.35)
 j=int(u);cmd=base.copy();cmd[:15]=arr[j]+(u-j)*(arr[j+1]-arr[j]);cmd[0]+=float(os.getenv('X','0'));cmd[1]+=float(os.getenv('Z','.049'))
 if mode=='pick':cmd[1]-=float(os.getenv('PULL','.015'))*smooth((t-.9)/.58);cmd[0]+=.000*smooth((t-1.5)/.36)
 s.clear_forces();q=s.joint_q.numpy();off=R.from_quat(q[-4:]).apply(tip);pos=q[-7:-4]+off;delta=anchor-pos;length=np.linalg.norm(delta);F=max(0,55*(length-L)+.18*(length-prev)/dt) if attached and length>L else 0;prev=length;event=attached and F>=threshold
 if event:attached=False
 if attached:
  force=F*delta/length;forces=s.body_f.numpy();forces[bi,:3]=force;forces[bi,3:]=np.cross(off,force);s.body_f.assign(forces)
 ctrl.joint_target_q.assign(cmd);v.step(s,n,ctrl,None,dt);s,n=n,s;loads=np.zeros(3);ext=0
 for h in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[h];gg=[gn[i] for i in co.geom];f=np.zeros(6);mj.mj_contactForce(mm,dd,h,f)
  if 'berry' not in gg or f[0]<1e-5:continue
  i=1-gg.index('berry');body=mm.geom(int(co.geom[i])).name.split('/')[-2];contacts.append([t,body,gg[i],*co.pos,f[0]]);ext+=not gg[i].startswith('handgeom');loads[0 if body=='link_1_3' else 1 if body=='link_2_3' else 2]+=f[0]
 q=s.joint_q.numpy().copy();bp=s.body_q.numpy()[palm];rel=R.from_quat(bp[3:]).inv().apply(q[-7:-4]-bp[:3]);rows.append([t,F,attached,event,*loads,ext,*rel,*q[-7:-4]]);qs.append(q)
a=np.array(rows);np.savez(out/(tag+'.npz'),a=a,q=qs)
for name,head,data in [('trace','t_s,tension_demand_N,attached,break_event,thumb_tip_N,index_tip_N,other_N,external_contacts,rel_x,rel_y,rel_z,x,y,z',rows),('contacts','t_s,link,geom,x,y,z,normal_N',contacts)]:
 with open(out/(tag+'_'+name+'.csv'),'w') as f:wr=csv.writer(f);wr.writerow(head.split(','));wr.writerows(data)
print(mode,a[::120,:8].round(3),a[:,1].max(),np.ptp(a[-360:,8:11],axis=0))
