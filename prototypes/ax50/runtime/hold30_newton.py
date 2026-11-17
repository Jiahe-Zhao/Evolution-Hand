import sys
from finger_skin import FINGER_SKIN_MM
from slim_layer import response
import argparse,hashlib
cli=argparse.ArgumentParser();cli.add_argument('task',choices=['pinch','release']);cli.add_argument('--x',type=int,default=8);cli.add_argument('--z',type=int,default=12);cli.add_argument('--arch',action='store_true');opt=cli.parse_args()
balltask=opt.task=='release';archmode='arch' if opt.arch else 'flat';thick=FINGER_SKIN_MM;import os;prefix=os.getenv('RUN','ob_pinch') if not balltask else os.environ['RUN']
import numpy as np,xml.etree.ElementTree as E,newton,warp as wp,json
from newton.sensors import SensorContact
from scipy.spatial.transform import Rotation
r=E.parse('hand.urdf');root=r.getroot()
# constrain wrist and finger posture, retaining all bodies and five palm hinges
for j in root.findall('joint'):
 if not j.get('name').startswith(('arch_','digit_')):j.set('type','fixed')
for j in root.findall('joint'):
 if j.get('name').startswith('digit_') and '_flex_' in j.get('name'):j.find('axis').set('xyz','-1 0 0')
def xml(s):root.append(E.fromstring(s))
xml('<link name="probe"><inertial><mass value=".12"/><inertia ixx=".00001" iyy=".00001" izz=".00001" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><sphere radius=".012"/></geometry></collision></link>')
xml('<joint name="probe_slide" type="prismatic"><parent link="diagnostic_world"/><child link="probe"/><origin xyz=".024 .034 .006"/><axis xyz="0 1 0"/><limit lower="-.04" upper=".02" effort="30" velocity=".2"/></joint>')
meta=np.load('sim.npz');P=meta['poses'][0];N=meta['names'].tolist();target=.45*P[N.index('link_1_0'),:3]+.55*P[N.index('link_1_1'),:3]
root.find("joint[@name='probe_slide']/origin").set('xyz',f'{target[0]} .034 {target[2]}')
j=root.find("joint[@name='arch_1']");parent=j.find('parent').get('link');origin=j.find('origin');xyz=origin.get('xyz');origin.set('xyz','0 0 0');j.find('parent').set('link','thumb_cmc_base')
xml('<link name="thumb_cmc_base"><inertial><mass value=".003"/><inertia ixx=".000001" iyy=".000001" izz=".000001" ixy="0" ixz="0" iyz="0"/></inertial></link>')
xml(f'<joint name="thumb_sweep" type="revolute"><parent link="{parent}"/><child link="thumb_cmc_base"/><origin xyz="{xyz}"/><axis xyz="0 -1 0"/><limit lower="{os.getenv('BACK_LIMIT','-.05')}" upper=".95" effort="1" velocity="2"/></joint>')
# One connected ellipsoid; root-guided heave plus elastic attachment to MC1.
xml('<link name="thenar"><inertial><mass value=".035"/><inertia ixx=".000003" iyy=".000005" izz=".000003" ixy="0" ixz="0" iyz="0"/></inertial></link>')
xml(f'<joint name="tissue" type="prismatic"><parent link="diagnostic_world"/><child link="thenar"/><origin xyz="{target[0]-.006} .006 {target[2]+.012}"/><axis xyz="0 1 0"/><limit lower="-.009" upper=".017" effort="80" velocity=".3"/></joint>')
# Separate the two overlapping metacarpal root cylinders and model CMC axial opposition.
jj=root.find("joint[@name='thumb_sweep']/origin");org=np.fromstring(jj.get('xyz'),sep=' ');org[2]-=.004;jj.set('xyz',' '.join(map(str,org)))
root.find("joint[@name='arch_1']/child").set('link','thumb_cmc_rollbase')
xml('<link name="thumb_cmc_rollbase"><inertial><mass value=".002"/><inertia ixx=".000001" iyy=".000001" izz=".000001" ixy="0" ixz="0" iyz="0"/></inertial></link>')
xml('<joint name="thumb_axial" type="revolute"><parent link="thumb_cmc_rollbase"/><child link="link_1_0"/><axis xyz="0 0 1"/><limit lower="-1.5" upper="1.5" effort=".8" velocity="2"/></joint>')
j=root.find("joint[@name='arch_1']/origin");j.set('rpy','1.6589451014157395 1.258485123922723 2.32124945343648')
j=root.find("joint[@name='thumb_sweep']/origin");j.set('xyz','0.015437572360192504 0.013100798997232026 -0.02433866249389827')
orig_rot=Rotation.from_euler('xyz',[1.6589451014157395,1.258485123922723,2.32124945343648])
mount_rot=Rotation.from_rotvec(np.deg2rad(np.loadtxt(os.environ['MOUNT_FILE'])))*orig_rot
root.find("joint[@name='arch_1']/origin").set('rpy',' '.join(map(str,mount_rot.as_euler('xyz'))))
root.find("joint[@name='thumb_sweep']/axis").set('xyz',os.getenv('SWEEP_AXIS','0 -1 0'));root.remove(root.find("joint[@name='probe_slide']"));root.remove(root.find("link[@name='probe']"));r.write(prefix+'.urdf')
b=newton.ModelBuilder(gravity=0);b.default_joint_cfg.armature=.00008
b.add_urdf(prefix+'.urdf',floating=False,collapse_fixed_joints=False,enable_self_collisions=True)
if balltask:
 start=np.fromstring(os.environ["BALL_MM"],sep=",")*.001; start[0]+=.065; radius=float(os.getenv("BALL_RADIUS","20"))*.001; mass=.035; inertia=.4*mass*radius**2;pb=b.add_body(xform=wp.transform(start.tolist(),wp.quat_identity()),mass=mass,inertia=wp.mat33(inertia,0,0,0,inertia,0,0,0,inertia),lock_inertia=True,label="probe");b.add_shape_sphere(pb,radius=radius,cfg=b.ShapeConfig(density=0,mu=.60),label="probe_geom");b.add_shape_box(-1,xform=wp.transform([0,-.075,0],wp.quat_identity()),hx=1.,hy=.005,hz=1.,label="catch_tray")
names=[x.split('/')[-1] for x in b.body_label];meta=np.load('sim.npz');pose=meta['poses'][0];mn=meta['names'].tolist();pads=[]
pads.append(b.add_shape_ellipsoid(names.index("thenar"),rx=.014,ry=.004,rz=.009,label="thenar_surface"))
for i in range(2,6):
 name=f'link_{i}_0';a=pose[mn.index(name)];end=pose[mn.index(f'link_{i}_1')][:3];pt=a[:3]+.55*(end-a[:3]);pt[1]=.001
 lp=Rotation.from_quat(a[3:]).inv().apply(pt-a[:3]);pads.append(b.add_shape_sphere(names.index(name),xform=wp.transform(lp.tolist(),wp.quat_identity()),radius=.008,label=f'pad_{i}'))
for i in range(5):
 for j in range(i):b.add_shape_collision_filter_pair(pads[i],pads[j])
# Embedded bases do not collide with their own tissue; distal fingers and ball still do.
for ii,body in enumerate(b.shape_body):
 if body>=0 and (names[body].endswith('_0') or ii in pads[1:]):b.add_shape_collision_filter_pair(pads[0],ii) if ii!=pads[0] else None
# Physical circumferential phalangeal sleeves incl. rounded endcaps. The palm covers metacarpals.
sleeves=[]
for di in range(1,6):
 for seg in range(0,3 if di==1 else 4):
  link=f'link_{di}_{seg}';col=root.find(f"link[@name='{link}']/collision");cyl=col.find('geometry/cylinder');length=float(cyl.get('length'));inner=float(cyl.get('radius'));cfg=b.ShapeConfig(density=0,ke=12000,kd=70,mu=.60,restitution=0)
  from dorsal_contact_mesh import shell
  verts,faces=shell(length,inner,seg)
  gid=b.add_shape_convex_hull(names.index(link),mesh=newton.Mesh(verts,faces,compute_inertia=False,maxhullvert=len(verts)),cfg=cfg,label=f'skin_{di}_{seg}')
  sleeves.append((gid,di,seg))
# Exclude embedded and immediately adjoining phalanges (continuous tissue across a hinge).
for gid,di,seg in sleeves:
 for jj,bb in enumerate(b.shape_body):
  if jj==gid or bb<0:continue
  nm=names[bb]
  if nm.startswith(f'link_{di}_') and abs(int(nm.split('_')[-1])-seg)<=1:b.add_shape_collision_filter_pair(gid,jj)
# A metacarpal shell is embedded in its own palm pad and in the basal tissue.
for gid,di,seg in sleeves:
 if seg==0:
  b.add_shape_collision_filter_pair(gid,pads[0])
  if di>=2:b.add_shape_collision_filter_pair(gid,pads[di-1])
# Neighboring metacarpal regions are the interior of the shared continuous palm.
# Exclude the complete body pair (required by the MuJoCo backend), not just the added sleeve.
for digit in (2,3,4):
 aa=names.index(f'link_{digit}_0');bb=names.index(f'link_{digit+1}_0')
 for i0,bd in enumerate(b.shape_body):
  if bd==aa:
   for j0,bd1 in enumerate(b.shape_body):
    if bd1==bb:b.add_shape_collision_filter_pair(i0,j0)
if balltask:
 probe_shape=next(i for i,bb in enumerate(b.shape_body) if bb==pb);b.add_shape_collision_filter_pair(pads[0],probe_shape)
m=b.finalize(device='cuda:0');js=[x.split('/')[-1] for x in m.joint_label];idx=lambda name:b.joint_qd_start[js.index(name)]
ids=[idx(z) for z in ['thumb_sweep','arch_1','digit_1_flex_1','digit_1_flex_2','digit_2_flex_1','digit_2_flex_2','digit_2_flex_3','thumb_axial','arch_2']]
ix=np.array([idx(f'arch_{i}') for i in range(1,6)]);ti=idx('tissue');yaw=ids[0];flex=ids[2:4];ik=np.load(os.getenv('GOAL','ob_goal.npy'));schedule=np.load('selected_schedule.npy'); t=np.arange(len(schedule))/500; u=lambda a,b:.5-.5*np.cos(np.pi*np.clip((t-a)/(b-a),0,1)); schedule=np.stack([u(a,b)*(1-u(4.4,6.5)) for a,b in [(1,3),(.8,3),(1,3),(1,3),(.6,2.7),(.6,2.7),(.6,2.7),(.8,2.8),(.7,2.8)]],axis=1)
geom=np.load('contour_surface.npz'); X,Z,Y,valid,ax,az,weights=[geom[h] for h in ('X','Z','Y','valid','ax','az','weights')]
ik=np.ones(9);schedule=np.load(os.environ["MOTION"])
def palm_dynamics(q,v,f):
 # Tend thenar support jointly from palm root and first metacarpal; rail remains at its original location.
 axis=np.array([-.70710678,0,.70710678]);sw=np.fromstring(os.getenv('SWEEP_AXIS','0 -1 0'),sep=' ');sw/=np.linalg.norm(sw)
 r0=Rotation.from_rotvec(sw*q[yaw])*mount_rot;r1=r0*Rotation.from_rotvec(axis*q[ix[0]]);r=r1*Rotation.from_rotvec([0,0,q[ids[7]]]);arm=r.apply([0,.001,.02475]);ind=[yaw,ix[0],ids[7]];J=np.cross(np.array([sw,r0.apply(axis),r1.apply([0,0,1])]),arm)[:,1];dy=arm[1]-0.0160500795675253
 load=850*(dy-q[ti])+2*(J@v[ind]-v[ti]);f[ti]=load-1650*q[ti]-4*v[ti];f[ind]-=(load-850*1650/2500*dy)*J
distids=np.array([idx(f'digit_{d}_flex_{2 if d==1 else 3}') for d in range(1,6)])
testdigit=int(os.getenv('DISTAL','0'))
flex_names=[f'digit_{di}_flex_{j}' for di in range(1,6) for j in range(1,3 if di==1 else 4)]
flex_ids=np.array([idx(j) for j in flex_names]);digits=np.load(os.environ['FLEX_MOTION']) if os.environ.get('FLEX_MOTION') else None
if digits is not None:assert digits.ndim==2 and digits.shape[1]==14
def wave(k):
 t=k*.002
 return np.deg2rad(45)*ramp(t,.5,2)*(1-ramp(t,3.4,5.4))
def drive(k,q,v):
 f=-1.3*q[:len(v)].copy()-.045*v; j=np.clip(k-(1000 if balltask else 0),0,len(schedule)-1);f[ids]=np.clip(2.8*(ik*schedule[j]-q[ids])-.055*v[ids],-.8,.8)
 if balltask and opt.arch:
  pre=ramp(k*.002,1.6,2.7);f[ix[0]]+=pre*.20*(.38-q[ix[0]]);f[ix[4]]+=pre*1.6*(.58-q[ix[4]])
 if digits is not None:
  f[flex_ids]=np.clip(2.8*(digits[min(k,len(digits)-1)]-q[flex_ids])-.055*v[flex_ids],-.8,.8)
 if testdigit:
  f=-1.3*q[:len(v)].copy()-.045*v;f[distids]=2.8*(-q[distids])-.055*v[distids];k0=distids[testdigit-1];f[k0]+=2.8*wave(k)
 palm_dynamics(q,v,f);return f
ramp=lambda t,a,b:.5-.5*np.cos(np.pi*np.clip((t-a)/(b-a),0,1))
meta=dict(urdf_sha=hashlib.sha256(open(prefix+'.urdf','rb').read()).hexdigest(),palm_sha=hashlib.sha256(open('contour_surface.npz','rb').read()).hexdigest(),finger_mm=thick,actuated_flex=sum(z.startswith('digit_') and 'flex' in z for z in js),hand_bodies=[z for z in names if z!='probe'])
open(prefix+'_model.json','w').write(json.dumps(meta))
if not balltask:
 shapeids=[i for i,body in enumerate(b.shape_body) if body>=0];sensor=SensorContact(m,sensing_shapes=shapeids,counterpart_shapes=shapeids);contacts=m.contacts();solver=newton.solvers.SolverMuJoCo(m);s=m.state();n=m.state();c=m.control();newton.eval_fk(m,m.joint_q,m.joint_qd,s)
 import mujoco
 mm=solver.mj_model;gm=[]
 for ii in range(mm.ngeom):
  label=mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_GEOM,ii) or ''
  mid=mm.geom_dataid[ii]
  if mid<0 or mm.geom_type[ii]!=mujoco.mjtGeom.mjGEOM_MESH:continue
  adr,num=mm.mesh_vertadr[mid],mm.mesh_vertnum[mid];rot=Rotation.from_quat(np.roll(mm.geom_quat[ii],-1));vtx=rot.apply(mm.mesh_vert[adr:adr+num])+mm.geom_pos[ii]
  gm.append([label,int(num),vtx.min(0).tolist(),vtx.max(0).tolist()])
 open(prefix+'_native_mesh.json','w').write(json.dumps(gm))
 table={gid:(d,k) for gid,d,k in sleeves};sg=[bb for bb,d,k in sleeves if (d,k) in [(1,2),(2,3)]];shape_names=[st.split('/')[-1] for st in b.shape_label];gi=lambda v:shapeids.index(v);poses=[];rows=[];contacts_audit={};ramp=lambda t,a,b:.5-.5*np.cos(np.pi*np.clip((t-a)/(b-a),0,1))
 for k in range(len(schedule)):
  t=k/500;q=s.joint_q.numpy();v=s.joint_qd.numpy();f=drive(k+1000,q,v); curl=ramp(t,1,3.8)*(1-ramp(t,8,9.5)); f[ix[4]]+=2.2*(.55*curl-q[ix[4]]); f[ix[3]]+=1.8*(.30*curl-q[ix[3]])
  for d in (3,4,5):
   for jj,ang in enumerate((62,64,27),1):
    joint=idx(f"digit_{d}_flex_{jj}");f[joint]=np.clip(2.8*(np.deg2rad(ang)*curl-q[joint])-.055*v[joint],-.8,.8);c.joint_f.assign(f);s.clear_forces();solver.step(s,n,c,None,.002);s,n=n,s
  solver.update_contacts(contacts,s);sensor.update(s,contacts);forces=sensor.force_matrix.numpy();positions=sensor.position_matrix.numpy();mat=np.linalg.norm(forces,axis=2);pose=s.body_q.numpy();poses.append(pose)
  ordering=list(sensor.sensing_indices.numpy() if hasattr(sensor.sensing_indices,'numpy') else sensor.sensing_indices)
  # counterpart indices are listed per row; resolve actual IDs for all pairs, including internal contacts.
  if k==0:
   counterpart=np.asarray(sensor.counterpart_indices.numpy() if hasattr(sensor.counterpart_indices,'numpy') else sensor.counterpart_indices);print('matrix',mat.shape,'indices',counterpart.shape,flush=True)
   pairs=[(i,j,int(u),int(vv)) for i,u in enumerate(ordering) for j,vv in enumerate(counterpart[i]) if j<mat.shape[1] and vv>=0 and u<vv]
   A,B,U,V=np.array(pairs).T;targetmask=(U==min(sg))&(V==max(sg));assert targetmask.sum()==1
   oi,oj=A[targetmask][0],B[targetmask][0]
  force=mat[oi,oj];cp=positions[oi,oj];pulp=[];ns=[];axial=[];cosines=[]
  for di,seg in [(1,2),(2,3)]:
   bpose=pose[names.index(f'link_{di}_{seg}')];r=Rotation.from_quat(bpose[3:]);ln=float(root.find(f"link[@name='link_{di}_{seg}']/collision/geometry/cylinder").get('length'));normal=r.apply([0,1,0]);ns.append(normal);pulp.append(bpose[:3]+r.apply([0,(3.2+thick)*.001,ln*.64]))
   local=r.inv().apply(cp-bpose[:3]);axial.append(local[2]/ln if force>1e-4 else np.nan);cosines.append(local[1]/max(np.linalg.norm(local[:2]),1e-10) if force>1e-4 else np.nan)
  for h in np.flatnonzero(mat[A,B]>.005):
   key=shape_names[U[h]]+' / '+shape_names[V[h]];old=contacts_audit.get(key,[0,0,0]);contacts_audit[key]=[max(old[0],float(mat[A[h],B[h]])),old[1]+1,old[2]+int(3.2<=t<=4.3)]
  rows.append([t,force,np.rad2deg(np.arccos(np.clip(ns[0]@ns[1],-1,1))),np.linalg.norm(pulp[0]-pulp[1])*1000,*axial,*cosines,np.max(mat[A[~targetmask],B[~targetmask]]),*np.rad2deg(q[ids]),*cp,*np.rad2deg(q[distids])])
 np.savetxt(prefix+'.csv',rows,delimiter=',',header='t,distal_N,normal_angle_deg,pulp_landmark_gap_mm,thumb_axial,index_axial,thumb_pulp_cos,index_pulp_cos,other_max_N,yaw,arch,mcp,ip,index_mcp,index_pip,index_dip,thumb_axial,index_arch')
 np.savez_compressed(prefix+'.npz',poses=poses,names=names)
 open(prefix+'_pairs.json','w').write(json.dumps(contacts_audit,indent=2));print(np.array(rows)[[0,350,-1]],flush=True)

 a=np.array(rows);hold=a[(a[:,0]>=3.3)&(a[:,0]<=4.2)]
 print("hold",hold[0],flush=True)

else:
 gids=[i for i,bb in enumerate(b.shape_body) if bb>=0 and names[bb]!="probe"]; allsensor=SensorContact(m,sensing_shapes=gids,counterpart_bodies=[pb]); mcshape=[ii for ii,bb in enumerate(b.shape_body) if bb==names.index('link_1_0')];mcSensor=SensorContact(m,sensing_shapes=mcshape,counterpart_bodies=[pb]);snapshots=[]; contacts=m.contacts();solver=newton.solvers.SolverMuJoCo(m);s=m.state();n=m.state();c=m.control();newton.eval_fk(m,m.joint_q,m.joint_qd,s)
 pr=idx("probe_free_joint");rows=[];states=[];frames=[];probe=pb
 lever=np.array([-L*np.cos(g) for L,g in zip([.045,.062,.066,.061,.052],np.deg2rad([135,110,108,111,116]))])*.55
 frames_out=[]
 w=np.zeros_like(Y); frames=[];states=[];hist=[];oldw=w.copy();bi=names.index('thenar');dt=.002
 goals=np.load(os.environ['CRADLE_Q']); ids5=[[idx(f'arch_{d}'),*[idx(f'digit_{d}_flex_{j}') for j in range(1,3 if d==1 else 4)]] for d in range(1,6)]
 for k in range(18001):
  t=k/500;q=s.joint_q.numpy();v=s.joint_qd.numpy();f=-1.3*q[:len(v)].copy()-.045*v
  for di,jj in enumerate(ids5):
   g=goals[di,:len(jj)].copy()
   if di>=3: phasew=np.array([ramp(t,.6,1.4),ramp(t,.3,1.3),ramp(t,.15,.95),ramp(t,.05,.75)])
   elif di==0:phasew=np.array([ramp(t,3.0,4.7),ramp(t,3.4,5.0),ramp(t,3.4,5.0)])
   else:phasew=np.array([ramp(t,2.4,3.8),ramp(t,2.8,4.6),ramp(t,2.5,4.0),ramp(t,2.4,3.8)])
   f[jj]=np.clip(2.8*(g*phasew-q[jj])-.055*v[jj],-.8,.8)
  for j0,val in [(ids[7],goals[0,3]),(yaw,goals[0,4])]:f[j0]=np.clip(2.8*(val*ramp(t,2.8,4.6)-q[j0])-.055*v[j0],-.8,.8)
  palm_dynamics(q,v,f)
  pose=s.body_q.numpy();ball=pose[probe,:3];pad=pose[bi,:3];local=(ball-pad)*1000; support=np.array([q[ti],*(lever[1:]*np.sin(q[ix[1:]]))]);sv=np.array([v[ti],*(lever[1:]*np.cos(q[ix[1:]])*v[ix[1:]])]);surface=Y+1000*(np.einsum('i,ijk->jk',support,weights)-q[ti]);dx=(local[0]-X);dz=(local[2]-Z);rr=(radius*1000)**2-dx*dx-dz*dz;ok=(rr>0)&valid;cap=np.sqrt(np.maximum(rr,.0001));lo=np.where(ok,np.maximum(surface+cap-local[1],0),0);oldw[:]=w;press=np.zeros_like(w)
  if np.any(lo):
   iz,ia=np.where(lo>0);sl=(slice(max(0,iz.min()-24),min(len(az),iz.max()+25)),slice(max(0,ia.min()-24),min(len(ax),ia.max()+25)));tmp=w[sl].copy();press[sl]=response(lo[sl].copy(),valid[sl].copy(),tmp);w[:]=0;w[sl]=tmp
  else:w[:]=0
  fy=press.sum()
  vel=s.body_qd.numpy()[probe];vb=vel[:3];omega=vel[3:];react=np.zeros(3);torque=np.zeros(3);slip=roll=cspeed=0.;cp=np.full(3,np.nan)
  if fy>1e-5:
   fn=np.array([(press*dx/cap).sum(),fy,(press*dz/cap).sum()]);norm=fn/np.linalg.norm(fn);cp=np.array([(press*X).sum(),(press*(surface-w)).sum(),(press*Z).sum()])/fy/1000+pad;arm=cp-ball
   share=np.einsum("jk,ijk->i",press,weights)/fy;vs=np.array([0,share@sv-(press*(w-oldw)).sum()/fy*.001/dt,0]);tangent=lambda a:a-np.dot(a,norm)*norm
   rel=tangent(vb+np.cross(omega,arm)-vs);slip=np.linalg.norm(rel);roll=np.linalg.norm(tangent(np.cross(omega,arm)));cspeed=np.linalg.norm(tangent(vb-vs))
   friction=-rel*min(.60*np.linalg.norm(fn)/max(slip,1e-12),1/(dt*(1/mass+np.dot(arm,arm)/inertia)));react=fn+friction;torque=np.cross(arm,friction) # spherical nodal normal forces pass through center
  grav=np.array([-mass*9.81,0.,0.]);assist=(1-ramp(t,5.3,6.0))*(1200*(start-np.array([.065*ramp(t,1.45,2.45),0,0])-ball)-12*vb-grav);force=react+grav+assist
  if fy>1e-5:
   f[ti]-=share[0]*react[1];f[ix[1:]]-=share[1:]*react[1]*lever[1:]*np.cos(q[ix[1:]])
  f[pr:pr+6]=np.r_[force,torque];c.joint_f.assign(f);s.clear_forces();solver.step(s,n,c,None,dt);s,n=n,s
  solver.update_contacts(contacts,s);allsensor.update(s,contacts);forces=allsensor.force_matrix.numpy()[:,0,:];order=np.asarray(allsensor.sensing_indices);group=np.zeros((6,3))
  for ii,gid in enumerate(order):
   name=names[b.shape_body[gid]];digit=int(name.split('_')[1]) if name.startswith(('link_1_','link_2_','link_3_','link_4_','link_5_','tip_')) else 0
   if name.startswith('link_') and name.endswith('_0'):digit=0
   group[digit]-=forces[ii]
  group[0]+=react
  rows.append([t,*ball,*vb,*assist,*group.flatten(),*np.rad2deg(q[ix]),*np.rad2deg(q[ids])])
  if k%10==0:states.append(pose.copy())
  if ball[0]<-.15:break
 np.savez_compressed(prefix+'.npz',rows=rows,poses=states,names=names,radius=radius,start=start)
