
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from make_long import *
import newton,sys,csv
mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode;o=d;pos=p.copy();start=tip.copy();paperz=start[2]-.0012-(.010 if mode=='hover' else 0);w.find("geom[@name='paper']").set('pos',fmt([start[0]+.020,start[1],paperz-.001]));
if os.getenv('TRI','1')=='1':from fit_tripod import fit
else:from fit_long import fit
control=fit(r,q);neutral=q[:27].copy()
if mode=='write':
 seed=np.load(o/'t4.npz');idx=1800;q=seed['q'][idx].copy()
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
if os.getenv('DP'):
 from squeeze_long import tighten
 # IK uses the pre-merge surface geometry but all runtime contacts use the single shaft.
 ikr=_evo_parse_xml(OUT/'pen140/t4.xml').getroot();control=tighten(ikr,q,control,float(os.getenv('DP','1.5')))
if mode=='write':
 snap=np.load(o/'mono1.npz');q=snap['q'][240].copy()
 for ar in snap['a'][:241]:
  off0[0]+=.006*(ar[2]-ar[4]);off0[2]-=.006*(ar[3]-ar[5]);off0[1]+=np.clip(.00008*(ar[9]-.035),-.00008,.00008);off0=np.clip(off0,[-.025,-.025,-.028],[.069,.025,.028])
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False);b.joint_q[:]=list(q)
for j,k in enumerate(b.joint_label):
 if 'diagnostic_wrist' in k.split('/')[-1]:i=b.joint_qd_start[j];b.joint_target_ke[i]=1700;b.joint_target_kd[i]=42
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=45);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=neutral;base[6:6+len(control)]=control;base[10]+=float(os.getenv('TH','0'));base[14]+=float(os.getenv('IN','0'));mm=v.mj_model;dd=v.mj_data;names=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];tip=names.index('tip');dt=1/960;xy=np.array([-.091,.003]);off=off0.copy();rows=[];qs=[];rear=[]
for t in range(4700 if mode=='write' else 3700):
 u=np.clip((t-500)/3960 if mode=='write' else (t-600)/3000,0,1);target=np.array([start[0]+.046*u,start[1]+.006*np.sin(2*np.pi*u)]);cmd=base.copy();cmd[:3]+=off;cmd[0]+=.046*u*(mode=='write');cmd[2]-=.006*np.sin(2*np.pi*u)*(mode=='write');ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,dt);s,n=n,s;xyz=dd.geom_xpos[tip];fn=0;gr=[0.,0.];contact=np.array([np.nan,np.nan])
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
 if mode=='hover':off[1]+=.0125*(start[2]-xyz[2])
 else:off[1]+=np.clip(float(os.getenv('GAIN','.00008'))*(fn-float(os.getenv('FORCE','.035'))),-.00008,.00008)
 off=np.clip(off,[-.025,-.025,-.028],[.069,.025,.028]);rows.append([t*dt,u,*target,*xyz,*contact,fn,*gr]);qs.append(s.joint_q.numpy().copy())
np.savetxt(o/(tag+'_origin.txt'),np.r_[start,paperz]);
with open(o/(tag+'_rear.csv'),'w') as ff:
 wr=csv.writer(ff);wr.writerow(['time_s','body','geom','force_N','x','y','z','gap_m','from_nib_mm','rod_segment']);wr.writerows(rear)
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs);print(mode,'stride',a[::240,[1,4,5,6,9,10,11]],flush=True)
