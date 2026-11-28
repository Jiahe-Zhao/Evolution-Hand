
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation as R
import os,sys,newton,csv
o=OUT/'hafted05_verify';o.mkdir(exist_ok=True);tag=sys.argv[1];bare=tag=='bare';dt=1/960;fmt=lambda a:' '.join(map(str,a))
r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');hand=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
src=_evo_parse_xml(OUT/'hafted05/asset.xml').getroot()
for a in src.find('asset'):r.find('asset').append(a)
ob=src.find('worldbody/body');ob.set('name','hammer');rot=R.from_euler('x',float(os.getenv('ROT','.98')));center=np.array([float(os.getenv('CX','-.012')),-.025,float(os.getenv('CZ','.085'))]);ob.set('pos',fmt(center-rot.apply([0,0,-float(os.getenv('GRIP','.085'))])));ob.set('quat',fmt(rot.as_quat()[[3,0,1,2]]))
parts=[g.get('name') for g in ob.findall('geom') if g.get('contype')!='0'];stone=[x for x in parts if x.startswith('stone_band')]
if not bare:w.append(ob)
for hg in hand.iter('geom'):
 for part in ([] if bare else parts)+['target']:
  E.SubElement(c,'pair',geom1=hg.get('name'),geom2=part,friction='.85 .85 .003 .0001 .0001',solref='.006 1')
for p in ([] if bare else stone):E.SubElement(c,'pair',geom1=p,geom2='target',solref='.003 1',solimp='.98 .995 .0001')
tx,ty,tz=[float(os.getenv(k,v)) for k,v in [('TX','-.012'),('TY','-.106'),('TZ','.054')]]
E.SubElement(w,'geom',name='target',type='ellipsoid',pos=fmt([tx,ty,tz]),size='.013 .013 .012',rgba='.67 .36 .15 1',contype='2',conaffinity='3')
E.SubElement(w,'geom',name='anvil',type='cylinder',pos=fmt([tx,ty,tz-.018]),size='.014 .006',rgba='.28 .30 .29 1',contype='2',conaffinity='3')
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i in range(3):b.joint_target_ke[i]=2500;b.joint_target_kd[i]=50
q=np.array(b.joint_q);q[6:11]=[.25,.7,.55,.65,.55];q[12:15]=[.80,.70,.50]
bend=float(os.getenv('BEND','1.3'))
for i in [16,20,24]:q[i:i+3]=[bend,bend,1.15]
desired=q.copy();warm=os.getenv('WARM')
if warm:q=np.load(o/(warm+'.npz'))['q'][int(os.getenv('WI','1920'))].copy()
if bare:q=q[:27]
b.joint_q[:]=list(q);m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=desired[:27];base[1]+=.025 if warm else 0;solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=70);mm=solver.mj_model;dd=solver.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];hids=[i for i,g in enumerate(gn) if g.startswith('handgeom')];target=gn.index('target');sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));rows=[];qs=[];mq=[];log=[]
pose=[]
for k in range(int(os.getenv('STEPS','4400'))):
 t=k*dt;cmd=base.copy();cmd[1]+=float(os.getenv('LIFT','.060'))*sm(t/.5);cmd[2]+=float(os.getenv('SHIFT','.050'))*sm((t-.65)/.45);cmd[1]+=(0 if warm else .025*sm((t-.1)/.4))-float(os.getenv('DOWN','.070'))*sm((t-3.3)/.11)*(1-sm((t-3.415)/.14));ctrl.joint_target_q.assign(cmd);s.clear_forces();solver.step(s,n,ctrl,None,dt);s,n=n,s;load=np.zeros(6);tip=np.zeros(5);F=H=external=depth=gd=wood=0
 for j in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[j];gg=[gn[i] for i in co.geom];ff=np.zeros(6);mj.mj_contactForce(mm,dd,j,ff)
  if ff[0]<1e-5:continue
  hi=next((i for i,x in enumerate(gg) if x.startswith('handgeom')),None);obj=any(x in parts for x in gg)
  if hi is not None and (obj or 'target' in gg):
   body=mm.geom(int(co.geom[hi])).name.split('/')[-2];other=gg[1-hi];log.append([t,body,other,*co.pos,ff[0],co.dist])
   if obj:
    ii=int(body.split('_')[1]) if body.startswith('link_') and body.split('_')[1] in list('12345') else 0;load[ii]+=ff[0];gd=max(gd,-co.dist)
    if ii and body.endswith('_3'):tip[ii-1]+=ff[0]
   else:H+=ff[0]
  if obj and 'target' in gg:
   if any(x in stone for x in gg):F+=ff[0];depth=max(depth,-co.dist)
   else:wood+=ff[0]
  if obj and any(x in ['floor','anvil'] for x in gg):external+=ff[0]
 gap=min(mj.mj_geomDistance(mm,dd,i,target,.3,None) for i in hids)*1000
 rows.append([t,F,H,external,wood,depth*1000,gd*1000,gap,*load,*tip]);qs.append(s.joint_q.numpy().copy());pose.append(s.body_q.numpy().copy());mq.append(dd.qpos.copy())
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs,mq=mq,pose=pose,names=b.body_label)
for suffix,header,data in [('trace','time_s,stone_target_N,hand_target_N,external_support_N,wood_target_N,impact_overlap_mm,grip_overlap_mm,hand_target_gap_mm,palm_N,thumb_N,index_N,middle_N,ring_N,little_N,thumb_tip_N,index_tip_N,middle_tip_N,ring_tip_N,little_tip_N',rows),('contacts','time_s,hand_link,object_part,x_m,y_m,z_m,normal_N,distance_m',log)]:
 with open(o/(tag+'_'+suffix+'.csv'),'w') as f:wr=csv.writer(f);wr.writerow(header.split(','));wr.writerows(data)
print(tag,a[:,1:8].max(0),a[::480,8:14].round(2),flush=True)
