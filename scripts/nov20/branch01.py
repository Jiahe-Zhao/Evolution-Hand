
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
import newton,csv,sys,os
from scipy.spatial.transform import Rotation as R
o=OUT/'branch01';o.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode
X=float(os.getenv('X','-.053'));Z=float(os.getenv('Z','.151'));load=2.;dt=1/480
r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');hand=w.find('body');hand.set('quat','.70710678 -.70710678 0 0');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
# No ground to catch a failed hand.
w.remove(w.find("geom[@name='floor']"));E.SubElement(w,'geom',name='branch',type='cylinder',fromto=f'{X} -.070 {Z} {X} .110 {Z}',size='.009',rgba='.52 .30 .13 1',contype='2',conaffinity='3')
ang=float(os.getenv('ANGLE','55'));rot=R.from_euler('y',ang,degrees=True);pivot=np.array([X,0,Z]);hand.set('pos',' '.join(map(str,pivot+rot.apply(np.array([0,0,.13])-pivot))));qq=(rot*R.from_euler('x',-90,degrees=True)).as_quat();hand.set('quat',' '.join(map(str,qq[[3,0,1,2]])))
for g in hand.iter('geom'):E.SubElement(c,'pair',geom1=g.get('name'),geom2='branch',friction='.8 .8 .001 .0001 .0001',solref='.009 1')
p=o/(tag+'.xml');E.ElementTree(r).write(p);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=1.;b.default_joint_cfg.target_kd=.04;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
for i in [0,2]:b.joint_target_ke[i]=1800;b.joint_target_kd[i]=40
free=os.getenv("UNGUIDED","1")=="1"; gains=[(b.joint_target_ke[i],b.joint_target_kd[i]) for i in range(6)]
if free:
 for i in range(6):b.joint_target_ke[i]=b.joint_target_kd[i]=0
b.joint_target_ke[1]=b.joint_target_kd[1]=0;m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=0;solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=solver.mj_model;dd=solver.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];palm=next(i for i,x in enumerate(b.body_label) if x.endswith('/link_0_0'));sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1));rows=[];pts=[];qs=[];ref=None
start=float(os.getenv("START",".85"));
for k in range(int((start+2.70)*480)):
 t=k*dt;q=s.joint_q.numpy();dq=s.joint_qd.numpy();cmd=base.copy();g=sm((t-.12)/.52)
 if mode=='grip':
  for j in (12,16,20,24):cmd[j:j+3]=g*np.array([.82,1.55,1.4])
  cmd[6:11]=g*np.array([.30,.25,.48,.68,.65])
 else:cmd[12:15]=g*np.array([.06,.02,0])
 # After 0.85 s absolutely no axial position/velocity controller.
 f=np.zeros(m.joint_dof_count);support=(1800*((-.001 if mode=='grip' else -float(os.getenv('TOUCH','.009')))*g-q[1])-38*dq[1])*(1-sm((t-.70)/.15)) if t<.85 else 0;F=load*sm((t-start)/.50);f[1]=support
 if free and t<.85:
  for i in (0,2,3,4,5):f[i]=(-gains[i][0]*q[i]-gains[i][1]*dq[i])*(1-sm((t-.70)/.15))
 s.clear_forces();bf=s.body_f.numpy();bf[palm,2]=-F;s.body_f.assign(bf);ctrl.joint_target_q.assign(cmd);ctrl.joint_f.assign(f);solver.step(s,n,ctrl,None,dt);s,n=n,s;q=s.joint_q.numpy().copy();bp=s.body_q.numpy()[palm,:3].copy()
 if ref is None or t<=.70:ref=bp.copy()
 nz=0.;forces=np.zeros(6);tips=np.zeros(5)
 for h in np.flatnonzero((dd.contact.dist<=0)&(dd.contact.efc_address>=0)):
  co=dd.contact[h];gg=[gn[i] for i in co.geom]
  if 'branch' not in gg:continue
  a=1-gg.index('branch');body=mm.geom(int(co.geom[a])).name.split('/')[-2];ff=np.zeros(6);mj.mj_contactForce(mm,dd,h,ff)
  if ff[0]<1e-5:continue
  fw=co.frame.reshape(3,3).T@ff[:3]*(1 if a==1 else -1);nz+=fw[2];j=int(body.split('_')[1]) if body.startswith('link_') and body.split('_')[1].isdigit() else 0;forces[j]+=ff[0]
  if j>0 and body.endswith('_3'):tips[j-1]+=ff[0]
  pts.append([t,body,gg[a],*co.pos,ff[0],*fw])
 rows.append([t,F,support,np.linalg.norm(bp-ref)*1000,nz,*forces,*tips,*bp,q[1]]);qs.append(q)
 if t>.70 and bp[2]<ref[2]-.105:break
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs)
for name,head,data in [('contacts','time_s,body,geom,x,y,z,normal_N,Fx,Fy,Fz',pts),('trace','time_s,added_load_N,axis_servo_N,slip_mm,branch_reaction_z_N,palm_N,thumb_N,index_N,middle_N,ring_N,little_N,thumb_tip_N,index_tip_N,middle_tip_N,ring_tip_N,little_tip_N,palm_x,palm_y,palm_z,slide_q',rows)]:
 with open(o/(tag+'_'+name+'.csv'),'w') as f:wr=csv.writer(f);wr.writerow(head.split(','));wr.writerows(data)
print(tag,'end',a[-1,:16].round(2),'peak touch',a[:,5:11].max(),'load slip',a[a[:,0]>=.85,3].max())
