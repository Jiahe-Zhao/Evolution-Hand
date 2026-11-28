
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
from pathlib import Path
import json,sys,os,newton,subprocess
out=OUT/'grind12';out.mkdir(exist_ok=True)
cx=-.093;cy=.003;z0=.020;cz=.0315;edges=np.array([-.032,-.018,-.006,.006,.018,.032]);levels=np.array([.003,.0010,0,0,.0010,.003])
def scene():
 r=_evo_parse_xml(OUT/'pebble.xml').getroot();w=r.find('worldbody');hand=w.find('body');a=r.find('asset')
 for obj in list(w):
  if obj.get('name','').startswith(('item','neighbor','rim','tray')):w.remove(obj)
 r.remove(r.find('contact'));con=E.SubElement(r,'contact')
 def pair(a,b,mu):E.SubElement(con,'pair',geom1=a,geom2=b,solref='.009 1',friction=f'{mu} {mu} .002 .0001 .0001')
 def slab(body,name,x,y,z,hx,hy,hz,slope,col,mass=None):
  angle=np.arctan(slope);geom(body,name,'box',(x,y+hz*np.sin(angle),z-hz*np.cos(angle)),f'{hx} {hy/np.cos(angle)} {hz}',col,euler=f'{angle} 0 0',**({'mass':str(mass)} if mass else {}))
 for i in range(5):
  y0,y1=edges[i:i+2];z1,z2=levels[i:i+2];slab(w,'bed'+str(i),cx+.020,cy+(y0+y1)/2,z0+(z1+z2)/2,.066,(y1-y0)/2,.006,(z2-z1)/(y1-y0),'.42 .35 .29 1')
 geom(w,'support','box',(cx+.020,cy,.002),'.063 .031 .002','.39 .32 .26 1')
 ob=E.SubElement(w,'body',name='stone',pos=f'{cx} {cy} {cz+.00015}');E.SubElement(ob,'freejoint',name='stone_free')
 # Three matching load-bearing facets; slight narrowing at the ends.
 for i,(y0,y1,z1,z2) in enumerate([(-.012,-.006,.0005,0),(-.006,.006,0,0),(.006,.012,0,.0005)]):
  angle=np.arctan((z2-z1)/(y1-y0));h=.002
  geom(ob,'sole'+str(i),'box',(0,(y0+y1)/2-h*np.sin(angle),z0-cz+(z1+z2)/2+h*np.cos(angle)),f'{.022 if i==1 else .019} {(y1-y0)/2/np.cos(angle)} {h}', '.36 .41 .39 1',euler=f'{angle} 0 0',mass=str(2500*(.034 if i==1 else .030)*(y1-y0)*2*h))
 # Distributed rounded asperities sample the finite contact footprint on each facet.
 for i,x in enumerate([-.015,-.005,.005,.015]):
  for j,y in enumerate([-.009,0,.009]):
   slope=np.sign(y)/12 if y else 0.;z=z0-cz+max(abs(y)-.006,0)/12
   geom(ob,f'sole_pad{i}_{j}','ellipsoid',(x,y,z+.001-.00045),'.0038 .0023 .001','.36 .41 .39 1',euler=f'{np.arctan(slope)} 0 0',mass='.000045')
 for side,x in enumerate([-.015,.0105]):
  geom(ob,'gripflare'+str(side),'ellipsoid',(x,0,-.001),'.0065 .015 .0075','.43 .46 .41 1',mass='.0077')
 geom(ob,'shoulder','ellipsoid',(-.002,0,-.0053),'.020 .012 .004','.36 .41 .39 1',mass='.0092')
 geom(ob,'grip','ellipsoid',(0,0,.0003),'.0105 .0074 .0084','.47 .49 .43 1',mass='.0068')
 for h in hand.iter('geom'):
  for g in ob.iter('geom'):pair(h.get('name'),g.get('name'),.95)
  for i in range(5):pair(h.get('name'),'bed'+str(i),.65)
 for g in ob.iter('geom'):
  for i in range(5):pair(g.get('name'),'bed'+str(i),.42)
 p=out/'grind12.xml';E.ElementTree(r).write(p);return p
def run(tag):
 p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);names=[k.split('/')[-1] for k in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};active=[]
 for i,k in enumerate(names):
  j=b.joint_qd_start[i]
  if 'diagnostic_wrist' in k:b.joint_target_ke[j]=1700;b.joint_target_kd[j]=42
  if k in valid and k.startswith(('wrist_','diagnostic_wrist_','link_')):active.append(j)
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();ix=[k.split('/')[-1] for k in m.body_label].index('stone')
 mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name for i in range(mm.ngeom)];src=np.load(OUT/'stack09/custom_actions.npy');qs=[];rows=[];actions=[]
 K=int(os.getenv('K','252'));T=int(os.getenv('T','200'));start=440;target=float(os.getenv('FN','1.6'));zadj=0;fy=0;fc=0;gain=float(os.getenv('GAIN','.000035'));shift=np.array([float(os.getenv('DX',0)),float(os.getenv('DZ',0)),float(os.getenv('DY',0))]);BEND=float(os.getenv('BEND',0))
 S=int(os.getenv('SUB','4'));dxfix=0.;dyfix=0.;scale=float(os.getenv('SCALE','1'));
 for raw in range((start+6*T+30)*S):
  t=raw/S
  act=src[int(min(t,K))].copy();a=np.clip((t-K)/70,0,1);act[:3]+=a*shift;act[10]+=a*BEND;act[14]+=a*BEND
  if t>K and t<start+6*T+30:zadj=np.clip(zadj+gain/S*(fc-target),-.015,.010)
  act[1]+=zadj;phase=np.clip((t-start)/T,0,6);act[0]+=.021*scale*(1-np.cos(np.pi*phase))
  cmd=base.copy();cmd[active]=act;ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,1/(240*S));s,n=n,s;pose=s.body_q.numpy()[ix];rot=R.from_quat(pose[3:]);tilt=np.rad2deg(np.arccos(np.clip(rot.as_matrix()[2,2],-1,1)))
  total=fore=aft=0.;side=set();finger=set();pts=[];spread=0.
  for j,co in enumerate(dd.contact):
   if co.dist>0 or co.efc_address<0:continue
   ge=[gn[k] for k in co.geom];parts=[g.split('/')[-2] for g in ge];ss=[g.split('/')[-1].rsplit('_',1)[0] for g in ge]
   if 'stone' not in parts:continue
   k=parts.index('stone');oth=1-k;f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
   if f[0]<1e-5:continue
   if parts[oth].startswith('link_'):finger.add(parts[oth].split('_')[1])
   if ss[k].startswith('sole') and ss[oth].startswith('bed'):
    xx,yy,zz=rot.inv().apply(co.pos-pose[:3]);total+=f[0];fore+=f[0]*(xx>0);aft+=f[0]*(xx<=0)
    if f[0]>.001:pts.append((xx,yy))
  if len(pts)>2:
   from scipy.spatial import ConvexHull,QhullError
   try:spread=ConvexHull(pts).volume*1e6
   except QhullError:pass
  if os.getenv('DEBUG') and raw>2000 and total<.01:
   for j,co in enumerate(dd.contact):
    ge=[gn[k] for k in co.geom];parts=[a.split('/')[-2] for a in ge]
    if 'stone' in parts:
     wrench=np.zeros(6);mj.mj_contactForce(mm,dd,j,wrench)
     if wrench[0]>.02:print([g.split('/')[-1] for g in ge],round(co.dist*1000,3),round(wrench[0],3))
   print('step',raw,pose);return
  fc=total;held='1' in finger and bool(finger.intersection({'2','3','4','5'}));rows.append([t/240,*pose,total,fore,aft,spread,tilt,held,zadj]);
  if raw%S==0:qs.append(s.joint_q.numpy().copy());actions.append(act)
 x=np.array(rows);np.savez_compressed(out/(tag+'.npz'),data=x,q=qs,actions=actions,names=np.array(names),starts=b.joint_q_start,start=start*S,T=T*S,S=S)
 for j in range(6):
  y=x[(start+j*T)*S:(start+(j+1)*T)*S+1];print(j,np.round([1000*(y[-1,1]-y[0,1]),y[:,8].min(),y[:,8].max(),y[:,13].mean(),y[:,11].min(),y[:,12].max(),y[:,9:11].min()],3))
if __name__=='__main__':run(sys.argv[1])
