
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
import os,sys,json,csv,re
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np,mujoco as mj,newton
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from bristles import build_bristles
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/brush14';fmt=lambda x:' '.join(map(str,x))
tag=sys.argv[1] if len(sys.argv)>1 else 'trial';mode=os.getenv('MODE','basic')
r=_evo_parse_xml(OUT/'pose.xml').getroot();w=r.find('worldbody');h=w.find('body');ob=w.find("body[@name='brush']");seed=np.load(OUT/os.getenv('POSE','grasp_seed.npz'));center=seed['center'];q0=seed['q'];handtarget=seed['control'];ob.set('pos',fmt(center))
root=E.SubElement(ob,'body',name='tuft_mount',pos='0 0 -.035');meta=build_bristles(root)
# Passive elastic hinges; all spring rest angles stay at zero throughout the rollout.
for j in root.iter('joint'):
 j.set('stiffness','0');j.set('damping','0');j.set('armature','0')
for hg in h.iter('geom'):hg.set('contype','1');hg.set('conaffinity','2')
ob.find('geom').set('contype','2');ob.find('geom').set('conaffinity','9')
for g in root.iter('geom'):
 g.set('contype','0' if g.get('name').endswith('_tip') else '4');g.set('conaffinity','0' if g.get('name').endswith('_tip') else '8')
paper_z=center[2]-.035-.022-.008
if mode=='air':paper_z-=.030
E.SubElement(w,'geom',name='paper',type='box',size='.065 .060 .001',pos=fmt([center[0]+.014,center[1],paper_z-.001]),rgba='.96 .94 .87 1',contype='8',conaffinity='7')
E.SubElement(w,'geom',name='table',type='box',size='.09 .08 .008',pos=fmt([center[0]+.014,center[1],paper_z-.010]),rgba='.23 .27 .29 1',contype='0',conaffinity='0')
E.SubElement(w,'light',pos='-.1 -.2 .5',dir='0 0 -1',diffuse='.7 .7 .7',ambient='.4 .4 .4')
c=E.SubElement(r,'contact')
for hg in h.iter('geom'):
 E.SubElement(c,'pair',geom1=hg.get('name'),geom2='shaft',condim='6',friction='1.1 1.1 .004 .004 .004',solref='.004 1',solimp='.95 .99 .001')
 E.SubElement(c,'pair',geom1=hg.get('name'),geom2='paper',condim='3',friction='.5 .5 .001 .0001 .0001',solref='.004 1')
for g in root.iter('geom'):
 if g.get('name').endswith('_tip'):continue
 E.SubElement(c,'pair',geom1=g.get('name'),geom2='paper',condim='3',friction='.45 .45 .00001 .00001 .00001',solref='.002 1',solimp='.95 .99 .0001')
capsules=[g.get('name') for g in root.iter('geom') if g.get('name').endswith('_capsule')]
if os.getenv('SELF','1')=='1':
 for i,g in enumerate(capsules):
  for g2 in capsules[i+1:]:
   if g.split('_')[1]!=g2.split('_')[1]:E.SubElement(c,'pair',geom1=g,geom2=g2,condim='3',friction='.1 .1 .00001 .00001 .00001',solref='.00025 1',solimp='.95 .99 .0001')
act=E.SubElement(r,'actuator')
for joint in h.iter('joint'):
 name=joint.get('name');slide='diagnostic_wrist' in name
 E.SubElement(act,'position',name='servo_'+name,joint=name,kp='2500' if slide else '10',kv='60' if slide else '.20')
path=OUT/(tag+'.xml');E.ElementTree(r).write(path)
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=3;b.default_joint_cfg.target_kd=.10;b.default_joint_cfg.armature=.0001;b.add_mjcf(str(path),collapse_fixed_joints=False);b.joint_q[:27]=q0.tolist()
# Use explicit physical EI/l spring constants rather than relying on MJCF spring import.
EI=float(os.getenv('EI','2e-6'));damp=float(os.getenv('DAMP','2e-7'));spring_dofs=[]
for j,label in enumerate(b.joint_label):
 name=label.split('/')[-1];i=b.joint_qd_start[j]
 if 'diagnostic_wrist' in name:b.joint_target_ke[i]=2500;b.joint_target_kd[i]=60
 if name.startswith('bristle_') or name.startswith('br_'):

  end=b.joint_qd_start[j+1] if j+1<len(b.joint_label) else len(b.joint_target_ke)
  for aidx in range(i,end):
   b.joint_target_ke[aidx]=0;b.joint_target_kd[aidx]=0;b.joint_armature[aidx]=0;spring_dofs.append(aidx)
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=60);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();base[:27]=handtarget;mm=solver.mj_model;dd=solver.mj_data
for ji in range(mm.njnt):
 if 'bristle_' in mm.joint(ji).name:
  mm.jnt_stiffness[ji]=EI/(.022/5);mm.dof_damping[mm.jnt_dofadr[ji]]=damp;mm.qpos_spring[mm.jnt_qposadr[ji]]=0
mm.opt.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST;mm.opt.cone=mj.mjtCone.mjCONE_ELLIPTIC;mm.opt.impratio=100
names=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];brush_ids=[i for i in range(mm.nbody) if 'brush' in mm.body(i).name];shaft=names.index('shaft');paper=names.index('paper');hair=[i for i,x in enumerate(names) if x not in ['shaft','paper','table'] and ('bristle' in x or x.startswith('br_'))];print('spring',len(spring_dofs),'hair geoms',len(hair),'bodies',mm.nbody,flush=True)
if not spring_dofs or not hair:raise RuntimeError('Missing imported passive hair joints/geometries')
# Newton MJCF import generates pair rows but drops authored condim/pair overrides.
# Set and audit the actual CPU solver contact model explicitly.
resolved=[]
for pi in range(mm.npair):
 i,k=int(mm.pair_geom1[pi]),int(mm.pair_geom2[pi])
 if shaft in (i,k) and any(names[g].startswith('handgeom') for g in (i,k)):
  mm.pair_dim[pi]=6;mm.pair_friction[pi]=[1.1,1.1,.004,.004,.004];mm.pair_solref[pi]=[.004,1];mm.pair_solreffriction[pi]=[.00025,1];mm.pair_solimp[pi]=[.95,.99,.001,.5,2]
 elif paper in (i,k) and any(g in hair for g in (i,k)):
  mm.pair_dim[pi]=3;mm.pair_friction[pi]=[.45,.45,.00001,.00001,.00001];mm.pair_solref[pi]=[.00025,1];mm.pair_solimp[pi]=[.95,.99,.0001,.5,2]
 elif i in hair and k in hair:
  mm.pair_dim[pi]=3;mm.pair_friction[pi]=[.1,.1,.00001,.00001,.00001];mm.pair_solref[pi]=[.00025,1];mm.pair_solimp[pi]=[.95,.99,.0001,.5,2]
 else:continue
 resolved.append({'pair':[names[i],names[k]],'dim':int(mm.pair_dim[pi]),'friction':mm.pair_friction[pi].tolist()})
(OUT/(tag+'_resolved_contacts.json')).write_text(json.dumps(resolved,indent=2))

brushbody=mm.geom_bodyid[shaft];masses=mm.body_mass[brush_ids];body0=dd.xpos[brushbody].copy();dt=float(os.getenv('DT',str(1/9600)));steps=int(float(os.getenv('SIMTIME','8'))/dt)
rows=[];qrows=[];inks=[];contacts=[];hairposes=[];velrows=[]
sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
def path_at(t):
 if t<1.2:return np.zeros(3),0
 if t<2.5:return np.array([0,0,-.010*sm((t-1.2)/1.3)]),1
 if t<3.1:return np.array([0,0,-.010]),2
 if t<5.1:return np.array([.018*sm((t-3.1)/2),0,-.010]),3
 if t<6.1:return np.array([.018,0,-.010+.016*sm(t-5.1)]),4
 if t<7.1:return np.array([.018,.012*sm(t-6.1),.006]),5
 return np.array([.018,.012,.006]),6
# Optional six-stroke writing challenge, attempted only after the basic physical tests.
write_anchor=center[:2].copy();write_z_start=0.;last_centroid=None;last_force=0.;xy_correction=np.zeros(2);z_correction=0.;stroke_id=-1;target_xy=center[:2].copy()
if mode=='hao':
 ref=json.loads((ROOT/'hao_reference.json').read_text());medians=[np.array(v,float) for v in ref['medians']];relative_curves=[(v-medians[0][0])*(.045/901) for v in medians]
 phases=[];cursor=0.
 def segment(duration,fun,phase,sid):
  global cursor
  phases.append((cursor,cursor+duration,fun,phase,sid));cursor+=duration
 segment(1.2,lambda u:np.zeros(3),0,-1)
 segment(1.3,lambda u:np.array([0,0,-.010*sm(u)]),1,0)
 segment(.6,lambda u:np.array([0,0,-.010]),2,0)
 for sid,curve in enumerate(relative_curves):
  if sid:
   prev=relative_curves[sid-1][-1].copy();first=curve[0].copy()
   segment(.45,lambda u,p=prev:np.r_[p,-.010+.016*sm(u)],4,sid-1)
   segment(.65,lambda u,p=prev,f=first:np.r_[p+(f-p)*sm(u),.006],5,sid)
   segment(.65,lambda u,f=first:np.r_[f,.006-.016*sm(u)],1,sid)
   segment(.45,lambda u,f=first:np.r_[f,-.010],2,sid)
  arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
  segment(arc[-1]/(float(os.getenv('FIRST_SPEED','.009')) if sid==0 else .016),lambda u,c=curve,v=arc:np.r_[[np.interp(sm(u)*v[-1],v,c[:,k]) for k in [0,1]],-.010],3,sid)
  segment(.35 if sid==0 else .25,lambda u,p=curve[-1].copy():np.r_[p,-.010],2,sid)
 last=relative_curves[-1][-1].copy();segment(.65,lambda u:np.r_[last,-.010+.016*sm(u)],4,5);segment(.8,lambda u:np.r_[last+np.array([.065,.02])*sm(u),.006],5,5);segment(.6,lambda u:np.r_[last+[.065,.02],.006],6,5)
 def path_at(t):
  global stroke_id
  for t0,t1,fun,phase,sid in phases:
   if t<t1:
    stroke_id=sid;return fun((t-t0)/(t1-t0)),phase
  stroke_id=5;return np.r_[last,.006],6
 steps=int(float(os.getenv('SIMTIME',str(cursor)))/dt)
mm.opt.timestep=dt;mj.mj_saveModel(mm,str(OUT/(tag+'_resolved.mjb')))
mj.mj_forward(mm,dd);restxyz=dd.geom_xpos[hair].copy();restR=dd.xmat[brushbody].reshape(3,3).copy();restlocal=(restxyz-dd.xpos[brushbody])@restR
simulation_failure=None;last_solver_time=-1.
ik=mj.MjData(mm);ikpad=[names.index('handgeom60'),names.index('handgeom80')];grip_track=[]
for step in range(steps):
 if os.getenv('TRACK','1')=='1' and step%round(.05/dt)==0 and step*dt>.25:
  for finger,pad in zip([15,19],ikpad):
   ik.qpos[:]=dd.qpos;ref=dd.qpos[finger:finger+4].copy();lo=np.maximum(mm.jnt_range[finger:finger+4,0]+1e-5,ref-.05);hi=np.minimum(mm.jnt_range[finger:finger+4,1]-1e-5,ref+.05)
   def residual(x):
    ik.qpos[finger:finger+4]=x;mj.mj_kinematics(mm,ik);dist=mj.mj_geomDistance(mm,ik,pad,shaft,.05,np.zeros(6));return np.r_[1000*(dist+.00015),.1*(x-ref)]
   fit=least_squares(residual,np.clip(ref,lo,hi),bounds=(lo,hi),diff_step=1e-4,max_nfev=25);handtarget[finger:finger+4]=fit.x;grip_track.append([step*dt,finger,float(residual(fit.x)[0])])
 t=step*dt;offset,phase=path_at(t)
 if mode=='hao':
  if abs(t-3.1)<dt*.6 and last_centroid is not None:
   write_anchor=last_centroid.copy();write_z_start=z_correction;(OUT/(tag+'_curves.json')).write_text(json.dumps([(v+write_anchor).tolist() for v in relative_curves]))
  target_xy=write_anchor+offset[:2]
  if phase in (1,2,3) and (last_force>1e-4 or phase in (2,3)):
   z_correction=np.clip(z_correction+dt*np.clip(.04*(last_force-.08),-.004,.004),-.002,.006)
  if phase==5:
   xy_correction*=np.exp(-8*dt);z_correction+=dt*8*(write_z_start-z_correction)
  if t>=3.1 and phase in (2,3) and last_centroid is not None and last_force>1e-4:
   velocity=12*(target_xy-last_centroid);velocity*=min(1,.015/max(np.linalg.norm(velocity),1e-12));xy_correction=np.clip(xy_correction+dt*velocity,-.015,.015)
  offset=offset.copy();offset[:2]+=xy_correction;offset[2]+=z_correction
 cmd=base.copy();cmd[:27]=handtarget;cmd[0]+=offset[0];cmd[1]+=offset[2];cmd[2]-=offset[1];ctrl.joint_target_q.assign(cmd);s.clear_forces();solver.step(s,n,ctrl,None,dt);s,n=n,s
 if dd.time<last_solver_time or not np.isfinite(dd.qpos).all():
  simulation_failure={'step':step,'scheduled_time_s':t,'solver_time_s':float(dd.time),'reason':'solver reset or non-finite state'};break
 last_solver_time=dd.time
 if step%max(1,round(1/(480*dt))):continue
 mj.mj_forward(mm,dd)
 Fp=np.zeros(3);Fh=np.zeros(3);gr=np.zeros(5);fpaperhand=0.;fshaftpaper=0.;min_gap=0.;inkpts=[]
 for ci,co in enumerate(dd.contact):
  if co.efc_address<0 or co.dist>0:continue
  i,k=map(int,co.geom);force=np.zeros(6);mj.mj_contactForce(mm,dd,ci,force)
  if force[0]<=0:continue
  world=co.frame.reshape(3,3).T@force[:3]
  if paper in (i,k) and any(x in hair for x in [i,k]):
   F=world if k!=paper else -world;Fp+=F;min_gap=min(min_gap,float(co.dist));inkpts.append(co.pos.copy());inks.append([len(rows),t,*co.pos,force[0]])
  if shaft in (i,k):
   other=k if i==shaft else i;bn=mm.body(mm.geom_bodyid[other]).name
   if 'link_' in bn:
    F=world if k==shaft else -world;Fh+=F
    match=re.findall(r'link_([1-5])_([0-3])',bn);part='link_'+'_'.join(match[-1]) if match else bn
    for finger in range(1,6):
     if part.startswith('link_'+str(finger)+'_'):gr[finger-1]+=force[0]
    contacts.append([t,part,names[other],force[0],*F,*co.pos,co.dist])
   if other==paper:fshaftpaper+=force[0]
  if paper in (i,k) and any(names[x].startswith('handgeom') for x in [i,k]):fpaperhand+=force[0]
 last_force=Fp[2];recent=[v for v in inks[-len(inkpts):]] if inkpts else []
 last_centroid=np.average(np.array(recent)[:,2:4],axis=0,weights=np.array(recent)[:,5]) if recent else None
 pos=dd.xpos[brushbody].copy();R=dd.xmat[brushbody].reshape(3,3);local=(dd.geom_xpos[hair]-pos)@R;deflect=np.linalg.norm(local-restlocal,axis=1);width=np.ptp(local[:,:2],axis=0)
 rows.append([t,phase,*offset,*pos,*Fp,*Fh,*gr,float(np.max(deflect)),*width,fpaperhand,fshaftpaper,min_gap,len(inkpts),np.degrees(np.arccos(np.clip(R[2,2],-1,1)))])
 if mode=='hao':rows[-1].extend([stroke_id,*target_xy,*(last_centroid if last_centroid is not None else [np.nan,np.nan]),z_correction])
 qrows.append(dd.qpos.copy());hairposes.append(dd.geom_xpos[hair].copy())
np.savetxt(OUT/(tag+'_grip_ik.csv'),grip_track,delimiter=',',header='time_s,finger_start_q,distance_error_mm',comments='')
np.savez_compressed(OUT/(tag+'.npz'),a=rows,q=qrows,ink=np.array(inks).reshape(-1,6),hair=hairposes,restlocal=restlocal,hair_ids=hair,hair_names=np.array([names[i] for i in hair]))
header='time_s,phase,cmd_x,cmd_y,cmd_z,brush_x,brush_y,brush_z,paper_fx,paper_fy,paper_fz,hand_fx,hand_fy,hand_fz,thumb_N,index_N,middle_N,ring_N,little_N,max_bend_m,spread_x_m,spread_y_m,hand_paper_N,shaft_paper_N,max_overlap_m,ink_contacts,tilt_deg'
if mode=='hao':header+=',stroke_id,target_x_m,target_y_m,ink_centroid_x_m,ink_centroid_y_m,force_control_offset_z_m'
np.savetxt(OUT/(tag+'_trace.csv'),rows,delimiter=',',header=header,comments='')
with open(OUT/(tag+'_contacts.csv'),'w') as f:
 wr=csv.writer(f);wr.writerow(['t','body','geom','normal_N','fx','fy','fz','x','y','z','gap']);wr.writerows(contacts)
(OUT/(tag+'_model.json')).write_text(json.dumps({'EI_Nm2_per_bundle':EI,'spring_Nm_rad':EI/(.022/5),'damping_Nms_rad':damp,'spring_dofs':len(spring_dofs),'hair_geoms':len(hair),'paper_z':paper_z,'mode':mode,'dt':dt,'rest_center':center.tolist(),'free_brush':True,'metadata':meta},indent=2,default=str))
(OUT/(tag+'_solver_status.json')).write_text(json.dumps({'failure':simulation_failure,'warning_counts':[int(v.number) for v in dd.warning],'recorded_duration_s':rows[-1][0]},indent=2))
a=np.array(rows);print('done',tag,'max paper',a[:,10].max(),'min four grips',a[min(100,len(a)-1):,14:18].min(axis=0),'bend mm',a[:,19].max()*1000,'tilt',a[:,26].max(),flush=True)
