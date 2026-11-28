
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation as R
import sys,newton,os,csv
o=OUT/'strike05_long';o.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');hand=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact');dt=1/960
def geom(b,n,tp,pos,size,col,**kw):return E.SubElement(b,'geom',name=n,type=tp,pos=pos,size=size,rgba=col,contype='2',conaffinity='3',**kw)
old=os.getenv('OLD')=='1'
if mode=='hit':
 ob=E.SubElement(w,'body',name='stone',pos='0 0 0');E.SubElement(ob,'freejoint',name='stone_free')
 src=_evo_parse_xml(OUT/'scan20/hand.xml').getroot().find(".//body[@name='scanner']")
 for source,new in [('grip','core'),('heel','poll'),('bridge','waist'),('housing','face')]:
  g=src.find(f"geom[@name='{source}']");g.set('name',new);
  if new=='face':g.set('type','ellipsoid')
  g.set('rgba','.41 .42 .39 1');g.set('contype','2');g.set('conaffinity','3');ob.append(g)
 if old:
  geom(ob,'nose','ellipsoid','-.049 .006 .069','.014 .013 .021','.44 .45 .40 1',mass='.019')
 else:
  geom(ob,'neck','capsule','-.052 .006 .071','.012 .024','.44 .45 .40 1',mass='.025',euler='0 1.52 0')
  geom(ob,'nose','ellipsoid','-.081 .006 .068','.019 .016 .019','.46 .47 .41 1',mass='.031')
for hg in hand.iter('geom'):
 for part in ((['core','poll','waist','face','nose']+([] if old else ['neck'])) if mode=='hit' else [])+['nut']:E.SubElement(c,'pair',geom1=hg.get('name'),geom2=part,friction='.95 .95 .002 .0001 .0001',solref='.006 1')
if mode=='hit' and not old:E.SubElement(c,'pair',geom1='nose',geom2='nut',solref='.0025 1',solimp='.98 .995 .0001')
cx,cy,cz=[float(os.getenv(k,v)) for k,v in [('X','-.050'),('Y','.017'),('Z','.025')]];geom(w,'nut','ellipsoid',f'{cx} {cy} {cz}','.012 .011 .012','.72 .39 .14 1')
geom(w,'anvil','cylinder',f'{cx} {cy} {cz-(.014 if old else .018)}','.012 .006','.31 .31 .30 1')
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i in range(3):b.joint_target_ke[i]=2500;b.joint_target_kd[i]=50
q=np.array(b.joint_q);q[6:11]=[.25,.7,.55,.65,.55];q[12:15]=[.80,.70,.50]
for i in [16,20,24]:q[i:i+3]=[1.05,1.05,1.]
warm=os.getenv('WARM','');initial=np.load(o/(warm+'.npz'))['q'][960] if warm else q
b.joint_q[:]=initial.tolist();m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=q[:27];v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=60);mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));rows=[];contacts=[];qs=[]
hids=[i for i,nm in enumerate(gn) if nm.startswith('handgeom')];target=gn.index('nut');anvil=gn.index('anvil');audit=[];mid=[];mqs=[];noseid=gn.index('nose') if mode=='hit' else None
for k in range(3200):
 t=k*dt;cmd=base.copy();cmd[2]-=(.028 if mode=='bare' else 0)*sm((t-.6)/.3);cmd[0]+=(float(os.getenv('BX','-.007')) if mode=='bare' else 0)*sm((t-.6)/.3);cmd[1]+=.024*(1 if warm else sm((t-.1)/.4))-float(os.getenv('DOWN','.067') if mode=='hit' else os.getenv('BARE','.080'))*sm((t-2.7)/.095)*(1-sm((t-2.797)/.11));ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,dt);s,n=n,s;loads=np.zeros(6);tips=np.zeros(5);F=H=ext=0.
 depth=gripdepth=0.;face="";other=0.
 for j in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[j];gg=[gn[ii] for ii in co.geom];ff=np.zeros(6);mj.mj_contactForce(mm,dd,j,ff)
  if ff[0]<1e-5:continue
  stone=any(x in (['core','poll','waist','face','nose']+([] if old else ['neck'])) for x in gg);hg=next((i for i,x in enumerate(gg) if x.startswith('handgeom')),None)
  if stone and 'nut' in gg:
   depth=max(depth,-co.dist);face='/'.join(gg);other+=ff[0] if 'nose' not in gg else 0
  if stone and hg is not None:gripdepth=max(gripdepth,-co.dist)
  if hg is not None:
   obj=gg[1-hg];body=mm.geom(int(co.geom[hg])).name.split('/')[-2]
   if stone or obj=='nut':
    contacts.append([t,body,gg[hg],obj,*co.pos,ff[0]])
    if stone:
     bb=body.split('_');ii=int(bb[1]) if len(bb)>1 and bb[1] in list('12345') else 0;loads[ii]+=ff[0]
     if ii and body.endswith('_3'):tips[ii-1]+=ff[0]
    else:H+=ff[0]
  if stone and 'nut' in gg:F+=ff[0]
  if stone and any(x in ['anvil','floor'] for x in gg):ext+=ff[0]
 def gap(tid):
  return min(mj.mj_geomDistance(mm,dd,i,tid,.2,None) for i in hids)*1000
 audit.append([t,gap(target),gap(anvil),1000*depth,1000*gripdepth,other,gap(noseid) if noseid is not None else 0]);mqs.append(dd.qpos.copy())
 q=s.joint_q.numpy().copy();qs.append(q);nose=dd.geom_xpos[gn.index('nose')].copy() if mode=='hit' else np.zeros(3);rows.append([t,F,H,ext,*loads,*tips,*nose])
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs,audit=audit,mq=mqs)
for suffix,header,data in [('trace','time_s,stone_target_N,hand_target_N,stone_external_N,palm,thumb,index,middle,ring,little,thumb_tip,index_tip,middle_tip,ring_tip,little_tip,nose_x,nose_y,nose_z',rows),('contacts','time_s,link,hand_geom,object_geom,x_m,y_m,z_m,normal_N',contacts)]:
 with open(o/(tag+'_'+suffix+'.csv'),'w') as fp:wr=csv.writer(fp);wr.writerow(header.split(','));wr.writerows(data)
print(mode,a[:,1:4].max(0),a[::480].round(3),flush=True)
