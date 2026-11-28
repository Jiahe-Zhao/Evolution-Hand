
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
import sys,json,csv,os,newton
o=OUT/'leaf04';o.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;fmt=lambda x:' '.join(map(str,x))
r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');h=w.find('body');h.set('quat','.70710678 -.70710678 0 0');h.set('pos','0 0 .065');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
def geom(b,n,k,pos,size,col,**kw):return E.SubElement(b,'geom',name=n,type=k,pos=fmt(pos),size=fmt(size),rgba=col,contype='2',conaffinity='3',**kw)
geom(w,'earth','box',[-.113,.009,.031],[.050,.035,.003],'.36 .25 .16 1')
geom(w,'food','box',[-.121,.009,.035],[.011,.016,.001],'.99 .55 .08 1')
E.SubElement(w,'camera',name='auditTop',pos='-.121 .009 .35',xyaxes='1 0 0 0 1 0',fovy='19')
X=-.067
for i,y in enumerate([-.001,.020]):
 z=.080+i*.0008
 geom(w,'base'+str(i),'box',[X+.002,y,.055],[.003,.0025,.020],'.36 .32 .19 1')
 geom(w,'root'+str(i),'capsule',[X+.002,y,z-.0036],[.0014,.0036],'.41 .32 .18 1')
 b=E.SubElement(w,'body',name='leaf'+str(i),pos=fmt([X,y,z]));E.SubElement(b,'joint',name='hinge'+str(i),type='hinge',axis='0 1 0',range='0 1.58',armature='.000001',damping='.00032')
 geom(b,'stalk'+str(i),'capsule',[-.011,0,0],[.0014,.010],'.36 .47 .18 1',euler='0 1.5707963 0',mass='.0004')
 # Convex lamina outline: short petiole to wide belly and a tapered tip.
 outline=[(-.012,0),(-.026,-.0065),(-.045,-.0101),(-.062,-.0101),(-.073,-.0067),(-.080,0),(-.071,.0068),(-.061,.0100),(-.044,.0100),(-.026,.0064)]
 verts=[(x,yv,zz) for zz in [-.001,.001] for x,yv in outline];mesh='bladeMesh'+str(i);E.SubElement(r.find('asset'),'mesh',name=mesh,vertex=fmt(np.ravel(verts)));geom(b,'blade'+str(i),'mesh',[0,0,0],[.001],['.17 .46 .17 1','.39 .61 .19 1'][i],mesh=mesh,mass='.0036')
 for v,(x,yv,xx,yy) in enumerate([(-.016,0,-.076,0),(-.031,0,-.045,.007),(-.031,0,-.045,-.007),(-.047,0,-.060,.007),(-.047,0,-.060,-.007)]):
  E.SubElement(b,'site',name=f'vein{i}_{v}',type='capsule',fromto=fmt([x,yv,.001,xx,yy,.001]),size='.00032',rgba='.53 .66 .26 1')
 for hg in h.iter('geom'):
  for g in b.findall('geom'):E.SubElement(c,'pair',geom1=hg.get('name'),geom2=g.get('name'),friction='.65 .65 .001 .0001 .0001',solref=os.getenv('CONTACT','.012 1.3'))
E.SubElement(c,'pair',geom1='blade0',geom2='blade1')
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);ns=[k.split('/')[-1] for k in b.joint_label];js=[b.joint_q_start[ns.index('hinge'+str(i))] for i in range(2)];ks=[b.joint_qd_start[ns.index('hinge'+str(i))] for i in range(2)]
for k in ks:b.joint_target_ke[k]=b.joint_target_kd[k]=0
for k in range(3):b.joint_target_ke[k]=1800;b.joint_target_kd[k]=42
q=np.array(b.joint_q);q[0]+=.16 if mode!='hand' else 0;b.joint_q[:]=q.tolist();m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ct=m.control();base=ct.joint_target_q.numpy().copy();base[:27]=q[:27];v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];dt=1/480;rows=[];qs=[];contacts=[];smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
for j in range(1776 if mode=='hand' else 1056):
 t=j*dt;cmd=base.copy();s.clear_forces();f=np.zeros(m.joint_dof_count,np.float32);q=s.joint_q.numpy();f[ks]=-float(os.getenv('SPR','.0014'))*q[js]-0.04*(q[js]-np.clip(q[js],0,1.58))
 if mode=='hand':
  for i,k in enumerate((12,16)):
   a=smooth((t-[.20,1.04][i])/.55)*(1-smooth((t-2.58)/.46))*float(os.getenv(['A','B'][i],'.88'));cmd[k:k+3]+=a*np.array([.025,1, .8])
 else:f[ks[int(mode)]]+=.0038*smooth((t-.15)/.4)*(1-smooth((t-1.1)/.2))
 ct.joint_f.assign(f);ct.joint_target_q.assign(cmd);v.step(s,n,ct,None,dt);s,n=n,s;load=np.zeros((2,3))
 for j2 in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[j2];gg=[gn[i] for i in co.geom];finger=next((i for i,g in enumerate(gg) if g.startswith('handgeom')),None)
  if finger is None:continue
  g=gg[1-finger]
  if not g.startswith(('blade','stalk')):continue
  body=mm.geom(int(co.geom[finger])).name.split('/')[-2];force=np.zeros(6);mj.mj_contactForce(mm,dd,j2,force)
  if force[0]<=.0005:continue
  i=int(g[-1]);load[i,0 if body=='link_2_3' else 1 if body=='link_3_3' else 2]+=force[0];contacts.append([t,i,body,gg[finger],g,*co.pos,force[0]])
 q=s.joint_q.numpy().copy();qs.append(q);rows.append([t,*np.rad2deg(q[js]),*load.ravel()])
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs);print(tag,a[::144].round(3))
with open(o/(tag+'_contacts.csv'),'w') as ff:
 cw=csv.writer(ff);cw.writerow(['time_s','leaf','finger_link','geom','leaf_part','x_m','y_m','z_m','normal_N']);cw.writerows(contacts)
