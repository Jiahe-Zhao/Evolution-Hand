
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from make_long import *
import newton,sys,csv
mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;o=OUT/'hao14';o.mkdir(exist_ok=True);pos=p.copy();start=tip.copy();paperz=start[2]-.0012-(.010 if mode=='hover' else 0);w.find("geom[@name='paper']").set('pos',fmt([start[0]+.020,start[1],paperz-.001]));
if os.getenv('TRI','1')=='1':from fit_tripod import fit
else:from fit_long import fit
control=fit(r,q);neutral=q[:27].copy()
if mode in ('write','hover'):
 seed=np.load(d/'t4.npz');idx=1800;q=seed['q'][idx].copy()
 old_a=seed['a'][:idx+1];off0=np.zeros(3)
 for ar in old_a:
  off0[0]+=.006*(ar[2]-ar[4]);off0[2]-=.006*(ar[3]-ar[5]);off0[1]+=np.clip(.00008*(ar[9]-.035),-.00008,.00008);off0=np.clip(off0,[-.025,-.025,-.028],[.069,.025,.028])
else:off0=np.zeros(3)
# Use a single uninterrupted collision cylinder in the verified simulation.
prior=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));I=prior.body('pen').id;E.SubElement(ob,'inertial',pos=fmt(prior.body_ipos[I]),quat=fmt(prior.body_iquat[I]),mass=str(prior.body_mass[I]),diaginertia=fmt(prior.body_inertia[I]))
front=ob.find("geom[@name='front']");front.set('fromto',fmt(np.r_[loc(-.023),loc(.110)]));front.set('mass','.0065');front.set('name','shaft');ob.remove(ob.find("geom[@name='rear']"))
for pair in list(c):
 if 'rear' in pair.attrib.values():c.remove(pair)
 elif pair.get('geom1')=='front':pair.set('geom1','shaft')
if mode in ('write','hover'):
 snap=np.load(d/'mono1.npz');q=snap['q'][240].copy()
 for ar in snap['a'][:241]:
  off0[0]+=.006*(ar[2]-ar[4]);off0[2]-=.006*(ar[3]-ar[5]);off0[1]+=np.clip(.00008*(ar[9]-.035),-.00008,.00008);off0=np.clip(off0,[-.025,-.025,-.028],[.069,.025,.028])
if os.getenv('DP'):
 from squeeze_long import tighten
 # IK uses the pre-merge surface geometry but all runtime contacts use the single shaft.
 ikr=_evo_parse_xml(OUT/'pen140/t4.xml').getroot();control=tighten(ikr,q,control,float(os.getenv('DP','1.5')))
w.find("geom[@name='paper']").set('size','.075 .065 .001');w.find("geom[@name='paper']").set('pos',fmt([start[0]+.013,start[1]-.018,paperz-.001]));p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=float(os.getenv("KP","2"));b.default_joint_cfg.target_kd=float(os.getenv("KD",".08"));b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);b.joint_q[:]=list(q)
for j,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k.split('/')[-1]:i=b.joint_qd_start[j];b.joint_target_ke[i]=1700;b.joint_target_kd[i]=42
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=45);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=neutral;base[6:6+len(control)]=control;base[10]+=float(os.getenv('TH','0'));base[14]+=float(os.getenv('IN','0'));mm=v.mj_model;dd=v.mj_data;names=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];tip=names.index('tip');dt=1/960;xy=np.array([-.091,.003]);off=off0.copy();rows=[];qs=[];rear=[]
import json
ref=json.loads((ROOT/'hao_reference.json').read_text());strokes=[np.array(x,float) for x in ref['medians']];scale=float(os.getenv('WIDTH','.045'))/901
anchor=strokes[0][0].copy();curves=[start[:2]+(x-anchor)*scale for x in strokes]
schedule=[];speed=float(os.getenv('SPEED','.009'))
def add(n,xy,phase,sid,height):
 for k in range(n):schedule.append([xy(k/(n-1)) if callable(xy) else xy,phase,sid,height(k/(n-1)) if callable(height) else height])
sm=lambda u:u*u*(3-2*u)
add(576,curves[0][0],0,0,0.)
for sid,curve in enumerate(curves):
 if sid:
  old=curves[sid-1][-1]
  add(384,old,2,sid-1,lambda u:.006*sm(u))
  add(960,lambda u:old+(curve[0]-old)*sm(u),3,sid,.006)
  add(576,curve[0],4,sid,lambda u:.006*(1-sm(u)))
  add(384,curve[0],0,sid,0.)
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
 count=max(960,int(arc[-1]/speed/dt))
 add(count,lambda u:np.array([np.interp(u*arc[-1],arc,curve[:,k]) for k in [0,1]]),1,sid,0.)
 add(192,curve[-1],0,sid,0.)
add(576,curves[-1][-1],2,5,lambda u:.006*sm(u))
(o/(tag+'_curves.json')).write_text(json.dumps([x.tolist() for x in curves]))
for t,(target,phase,sid,height) in enumerate(schedule[:int(os.getenv("MAXSTEPS",str(len(schedule))))]):
 if os.getenv("TRACKGRIP") and t%96==0:
  live=s.joint_q.numpy().copy();updated=tighten(ikr,live,live[6:19].copy(),float(os.getenv("TRACKGRIP")));base[6:15]=updated[:9]
 target=np.array(target);u=t/max(1,len(schedule)-1);cmd=base.copy();cmd[6:19]=q[6:19]+min(1,t/480)*(base[6:19]-q[6:19]);cmd[:3]+=off;cmd[0]+=target[0]-start[0];cmd[2]-=target[1]-start[1];cmd[1]+=height;ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,dt);s,n=n,s;xyz=dd.geom_xpos[tip];fn=0;gr=[0.,0.];contact=np.array([np.nan,np.nan])
 for j,co in enumerate(dd.contact):
  if co.dist>0 or co.efc_address<0:continue
  i,k=co.geom;gg={names[i],names[k]};f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
  if gg=={'tip','paper'} and f[0]>0:fn+=f[0];contact=co.pos[:2].copy()
  if gg.intersection({'shaft'}) and f[0]>.0001:
   z=next(z for z in (i,k) if names[z] !='shaft');ax=dd.geom_xpos[names.index('shaft')]-xyz;ax=ax/np.linalg.norm(ax);rear.append([t*dt,mm.geom(z).name.split('/')[-2],names[z],f[0],*co.pos,co.dist,1+1000*np.dot(co.pos-xyz,ax),('rear' if 1+1000*np.dot(co.pos-xyz,ax)>=30 else 'front')])
  if gg.intersection({'shaft'}):
   for z in (i,k):
    part=mm.geom(z).name.split('/')[-2];
    if part.startswith(('link_1_','link_2_')):gr[int(part[5])-1]+=f[0]
 off[0]+=.006*(target[0]-xyz[0]);off[2]-=.006*(target[1]-xyz[1])
 if mode=='hover':off[1]+=.0125*(start[2]+height-xyz[2])
 elif phase in (0,1):off[1]+=np.clip(float(os.getenv('GAIN','.00008'))*(fn-float(os.getenv('FORCE','.035'))),-.00008,.00008)
 off=np.clip(off,[-.025,-.025,-.028],[.069,.025,.028]);rows.append([t*dt,u,*target,*xyz,*contact,fn,*gr,sid,phase,height]);qs.append(s.joint_q.numpy().copy())
np.savetxt(o/(tag+'_origin.txt'),np.r_[start,paperz]);
with open(o/(tag+'_rear.csv'),'w') as ff:
 wr=csv.writer(ff);wr.writerow(['time_s','body','geom','force_N','x','y','z','gap_m','from_nib_mm','rod_segment']);wr.writerows(rear)
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs);print(mode,'stride',a[::240,[1,4,5,6,9,10,11]],flush=True)
