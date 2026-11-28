
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation as R
import newton,sys,os
o=OUT/'pour21';o.mkdir(exist_ok=True);tag=sys.argv[1];fmt=lambda a:' '.join(map(str,a))
r=_evo_parse_xml(OUT/'scan20/hand.xml').getroot();w=r.find('worldbody');hand=w.find('body');cup=w.find("body[@name='scanner']");cup.set('name','cup');cup.remove(cup.find('body'));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
for g in list(w.findall('geom')):
 if g.get('name')!='floor':w.remove(g)
for g in list(cup.findall('geom')):
 if g.get('name') not in ('grip','heel','bridge','housing'):cup.remove(g)
for g in cup.findall('geom'):
 g.set('contype','2');g.set('conaffinity','7')
C=np.array([-.066,.006,.079]);A=.018;B=.016;H=.018;zc=.002
def geom(b,n,tp,pos,size,mass,col,**kw):
 return E.SubElement(b,'geom',name=n,type=tp,pos=fmt(pos),size=fmt(size),mass=str(mass),rgba=col,contype='2',conaffinity='7',friction='.38 .002 .0001',**kw)
def box(b,n,pos,size,mass,col):return geom(b,n,'box',pos,size,mass,col)
col='.17 .51 .55 1';lip='.95 .64 .17 1'
box(cup,'cupbase',C-[0,0,.001],[A+.002,B+.002,.001],.003,col)
for k in (-1,1):
 box(cup,'cupX'+str(k),C+[k*(A+.001),0,H/2],[.001,B+.002,H/2],.001,col)
 box(cup,'cupY'+str(k),C+[0,k*(B+.001),H/2],[A,.001,H/2],.001,col)
 for ax in [0,1]:
  off=np.array([k*(A+.001),0,H] if ax==0 else [0,k*(B+.001),H]);g=geom(cup,'rim'+str(ax)+str(k),'capsule',C+off,[.001,(B+.001 if ax==0 else A+.001)],.0002,lip,euler=('1.5707963 0 0' if ax==0 else '0 1.5707963 0'))
# bowl is shallow with sloping polygon walls and wide base
BC=np.array([-.066,-.057,.012]);rad=.054;nr=16
geom(w,'bowlbase','cylinder',BC-[0,0,.002],[rad,.002],1,'.28 .42 .59 1')
for j in range(nr):
 a=j*2*np.pi/nr;x,y=np.cos(a),np.sin(a);rot=R.from_euler('z',a)*R.from_euler('y',.28)
 geom(w,'bowlwall'+str(j),'box',BC+[(rad+.003)*x,(rad+.003)*y,.011],[.0018,.012,.0125],.01,'.27 .54 .65 1',quat=fmt(rot.as_quat()[[3,0,1,2]]))
for j in range(30):
 x=(j%6-2.5)*.0057;y=(j//6-2)*.0059;g=E.SubElement(w,'body',name=f'grain{j}',pos=fmt(C+[x,y,.0036]));E.SubElement(g,'freejoint',name=f'gfree{j}')
 E.SubElement(g,'inertial',mass='.00005',pos='0 0 0',diaginertia='8.84e-11 1.025e-10 1.109e-10');geom(g,f'seed{j}','ellipsoid',[0,0,0],[.0025,.0022,.0020],.00005,'.86 .56 .21 1',euler=f'0 0 {j*.51}')
for hg in hand.iter('geom'):
 for cg in cup.iter('geom'):E.SubElement(c,'pair',geom1=hg.get('name'),geom2=cg.get('name'),friction='1 1 .003 .0001 .0001',solref='.009 1')
# Explicit physical pairs for grains; retain actual hand-to-grain collisions.
for j in range(30):
 for g in [*cup.iter('geom'),*w.findall('geom')]+[w.find(f"body[@name='grain{i}']/geom") for i in range(j)]:
  E.SubElement(c,'pair',geom1=f'seed{j}',geom2=g.get('name'),condim=('6' if g.get('name').startswith('bowl') else '3'),friction=('.36 .36 .0007 .0002 .0002' if g.get('name').startswith('bowl') else '.36 .36 .0007 .0001 .0001'),solref='.005 1')
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);names=[n.split('/')[-1] for n in b.joint_label];js=lambda n:b.joint_q_start[names.index(n)];jcup=js('scan_free');jg=[js('gfree'+str(i)) for i in range(30)]
for i in range(3):b.joint_target_ke[i]=2500;b.joint_target_kd[i]=50
b.joint_target_ke[3]=8;b.joint_target_kd[3]=.4
q=np.array(b.joint_q);q[6:11]=[.25,.7,.55,.65,.55];q[12:15]=[.80,.70,.50]
for i in [16,20,24]:q[i:i+3]=[1.05,1.05,1.]
b.joint_q[:]=list(q);m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=q[:27];mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)]
dt=1/480;angle=float(os.getenv('ANG','70'));dwell=float(os.getenv('HOLD','.15'));rows=[];qs=[];locals_=[];xyzs=[];ex=np.zeros(30,bool);touch=[];seq=[];flags=[];stop=None;cut=int(os.getenv('CUT','31'));adj=np.zeros(2);meas=np.zeros(2);cl=lambda u:np.clip(u,0,1);smooth=lambda u:cl(u)**2*(3-2*cl(u))
handlelog=[];rimlog=[]
for k in range(int(os.getenv('STEPS','1728'))):
 t=k*dt;cmd=base.copy();phase=smooth((t-.55)/.85)*(1-smooth((t-1.40-dwell)/.48));
 if stop is None and ex.sum()>=cut:stop=(t,phase)
 if stop is not None:phase=stop[1]*(1-smooth((t-stop[0])/float(os.getenv('RETURN','.25'))))
 cmd[3]+=np.deg2rad(angle)*phase;cmd[1]+=.051*smooth((t-.14)/.35)
 # rotate about cup center via commanded wrist translation
 rot=R.from_rotvec([np.deg2rad(angle)*phase,0,0]);delta=C-[0,0,.13];shift=delta-rot.apply(delta);cmd[:3]+=shift[[0,2,1]]*[1,1,-1]
 if t>2.02:adj+=.008*np.array([-meas[0],meas[1]]);adj=np.clip(adj,-.55,.55)
 cmd[3]+=adj[0];cmd[5]+=adj[1];ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,dt);s,n=n,s;q=s.joint_q.numpy();cp=q[jcup:jcup+3];rr=R.from_quat(q[jcup+3:jcup+7]);xyz=np.array([q[j:j+3] for j in jg]);loc=rr.inv().apply(xyz-cp)-C
 outside=(abs(loc[:,0])>A+.002)|(abs(loc[:,1])>B+.002);new=outside&~ex;ex|=new;on=np.zeros(30,bool);rim=np.zeros(30,bool);handseed=0;grip=set();ext=0
 for j in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[j]
  if co.dist>0 or co.efc_address<0:continue
  ids=list(co.geom);gg=[gn[i] for i in ids];body=[mm.geom(i).name.split('/')[-2] for i in ids];f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
  if f[0]<=1e-5:continue
  if any(x.startswith('seed') for x in gg):
   si=next(i for i,x in enumerate(gg) if x.startswith('seed'));idx=int(gg[si][4:]);other=gg[1-si];on[idx]|=other.startswith('bowl');rim[idx]|=other.startswith('rim');handseed+=other.startswith('handgeom')
   if other.startswith('rim'):rimlog.append((k,t,idx,other,f[0]))
  if 'cup' in body:
   xx=1-body.index('cup');g=gg[xx]
   if g.startswith('handgeom'):
    grip.add(body[xx]);
    if gg[body.index('cup')] in ('grip','heel','bridge','housing'):handlelog.append((k,t,body[xx],gg[body.index('cup')],f[0]))
   if g.startswith(('bowl','floor')):ext+=1
 inb=(np.linalg.norm(xyz[:,:2]-BC[:2],axis=1)<rad+.004)&(xyz[:,2]>=BC[2]-.001)&(xyz[:,2]<BC[2]+.024)
 meas=rr.as_euler('xyz')[:2];tilt=np.degrees(np.arccos(np.clip(rr.as_matrix()[2,2],-1,1)));rows.append([t,tilt,ex.sum(),inb.sum(),handseed,len(grip),ext,rim.sum(),*(cp+rr.apply(C))]);seq.append(ex.copy());flags.append([on,rim]);locals_.append(loc);xyzs.append(xyz);qs.append(q.copy())
 if k>=1728 and ex.sum()>0 and np.max(np.linalg.norm(np.array(xyzs[-240:])[:,ex]-xyz[ex],axis=2))<.0008 and tilt<1:break
import csv
for suf,lab,rec in [('grip','step,time_s,link,part,normal_N',handlelog),('rim','step,time_s,grain_id,rim_part,normal_N',rimlog)]:
 with open(o/(tag+'_'+suf+'.csv'),'w') as f:
  wr=csv.writer(f);wr.writerow(lab.split(','));wr.writerows(rec)
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs,loc=locals_,xyz=xyzs,ex=seq,flags=flags)
inds=np.flatnonzero(np.diff(a[:,2],prepend=0)>0);print(tag,'exit',[(round(a[i,0],2),int(a[i,2])) for i in inds], 'tilt',a[:,1].max(),a[-1,:],'touch/external',a[:,4].max(),a[:,6].max(),'grip',a[:,5].min(),'min exit z',min((np.array(locals_)[i,np.array(seq)[i]&~np.array(seq)[max(0,i-1)],2].min() for i in inds),default=0))
