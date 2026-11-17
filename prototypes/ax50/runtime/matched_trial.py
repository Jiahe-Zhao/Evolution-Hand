import sys
from finger_skin import FINGER_SKIN_MM
from slim_layer import response
import argparse,hashlib
cli=argparse.ArgumentParser();cli.add_argument('task',choices=['pinch','release']);cli.add_argument('--x',type=int,default=8);cli.add_argument('--z',type=int,default=12);cli.add_argument('--arch',action='store_true');opt=cli.parse_args()
balltask=opt.task=='release';archmode='arch' if opt.arch else 'flat';thick=FINGER_SKIN_MM;import os;prefix=os.getenv('RUN','ob_pinch') if not balltask else f'ih_{archmode}_{opt.x}'
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
root.find("joint[@name='thumb_sweep']/axis").set('xyz',os.getenv('SWEEP_AXIS','0 -1 0'));root.remove(root.find("joint[@name='probe_slide']"));root.remove(root.find("link[@name='probe']"));C=np.load('under_center.npy');xml('<link name="fixed_ball"><inertial><mass value=".035"/><inertia ixx=".0000056" iyy=".0000056" izz=".0000056" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><sphere radius=".02"/></geometry></collision></link>');xml('<joint name="ball_fixture" type="fixed"><parent link="diagnostic_world"/><child link="fixed_ball"/><origin xyz="'+ ' '.join(map(str,C)) +'"/></joint>');side=os.environ['SIDE'];from matched_mirror import mirrored,L,W
if side=='right':mirrored(root);C=W@C
root.set('name','ax50_'+side);r.write(prefix+'.urdf')
b=newton.ModelBuilder(gravity=0);b.default_joint_cfg.armature=.00008
b.add_urdf(prefix+'.urdf',floating=False,collapse_fixed_joints=False,enable_self_collisions=True)
if balltask:
 start=np.array([target[0],.034,target[2]]);off=None;start[[0,2]]=np.array([opt.x,opt.z])*.001;pb=b.add_body(xform=wp.transform(start.tolist(),wp.quat_identity()),mass=.12,inertia=wp.mat33(.00001,0,0,0,.00001,0,0,0,.00001),lock_inertia=True,label="probe");b.add_shape_sphere(pb,radius=.012,label="probe_geom");b.add_shape_box(-1,xform=wp.transform([0,-.075,0],wp.quat_identity()),hx=1.,hy=.005,hz=1.,label="catch_tray")
names=[x.split('/')[-1] for x in b.body_label];meta=np.load('sim.npz');pose=meta['poses'][0];mn=meta['names'].tolist();pads=[]
pads.append(b.add_shape_ellipsoid(names.index("thenar"),rx=.014,ry=.004,rz=.009,label="thenar_surface"))
for i in range(2,6):
 name=f'link_{i}_0';a=pose[mn.index(name)];end=pose[mn.index(f'link_{i}_1')][:3];pt=a[:3]+.55*(end-a[:3]);pt[1]=.001
 lp=Rotation.from_quat(a[3:]).inv().apply(pt-a[:3]);pads.append(b.add_shape_sphere(names.index(name),xform=wp.transform((L@lp if side=="right" else lp).tolist(),wp.quat_identity()),radius=.008,label=f'pad_{i}'))
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
  if side=="right":verts=verts@L.astype(np.float32);faces=faces.reshape(-1,3)[:,::-1].copy().flatten()
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
bg=next(i for i,bb in enumerate(b.shape_body) if bb==names.index('fixed_ball'))
gids=[i for i,bb in enumerate(b.shape_body) if bb>=0 and i!=bg]
sensor=SensorContact(m,sensing_shapes=gids,counterpart_shapes=[bg]);ct=m.contacts();solver=newton.solvers.SolverMuJoCo(m);s=m.state();n=m.state();ctrl=m.control();newton.eval_fk(m,m.joint_q,m.joint_qd,s)
ixs=[g for g in sensor.sensing_indices];labels=[names[b.shape_body[g]] for g in ixs];rows=[];poses=[];tip=names.index('link_1_2')
for k in range(len(schedule)):
 q=s.joint_q.numpy();v=s.joint_qd.numpy();f=drive(k,q,v)
 if os.getenv('ADD_TWO'):
  extra=np.load(os.environ['EXTRA_GOALS']) if k==0 else extra
  for di in [2,3]:
   js0=[idx('arch_'+str(di))]+[idx(f'digit_{di}_flex_{a}') for a in (1,2,3)]
   f[js0]=np.clip(2.8*(extra[di-2]* (np.array([ramp(k/500,3.95,4.65),ramp(k/500,4.60,5.65),ramp(k/500,4.25,5.35),ramp(k/500,4.25,5.4)]) if di==2 else ramp(k/500,5.65,6.8))-q[js0])-.055*v[js0],-.8,.8)
 ctrl.joint_f.assign(f);s.clear_forces();solver.step(s,n,ctrl,None,.002);s,n=n,s;solver.update_contacts(ct,s);sensor.update(s,ct);fs=-sensor.force_matrix.numpy()[:,0,:];ps=sensor.position_matrix.numpy()[:,0,:];pose=s.body_q.numpy();tot=np.zeros((6,3));dist=np.zeros(3);pts=np.zeros(3);w=0;pw=np.zeros(6);pp=np.zeros((6,3))
 for ii,name in enumerate(labels):
  digit=int(name.split('_')[1]) if name.startswith(('link_1_','link_2_','link_3_','link_4_','link_5_','tip_')) else 0;tot[digit]+=fs[ii]
  if np.linalg.norm(fs[ii])>1e-5:
   pw[digit]+=np.linalg.norm(fs[ii]);pp[digit]+=np.linalg.norm(fs[ii])*ps[ii]
  if name=='link_1_2':
   mag=np.linalg.norm(fs[ii]);dist+=fs[ii]
   if mag>1e-5:pts+=mag*ps[ii];w+=mag
 cp=pts/w if w else np.full(3,np.nan);local=Rotation.from_quat(pose[tip,3:]).inv().apply(cp-pose[tip,:3])*1000
 rows.append([k/500,*cp,*dist,*local,*tot.flatten(),*np.rad2deg(q[ids]),*(pp[2]/pw[2] if pw[2] else [np.nan]*3),*(pp[3]/pw[3] if pw[3] else [np.nan]*3)])
 if k%10==0:poses.append(pose.copy())
np.savez_compressed(prefix+'.npz',poses=poses,names=names,rows=rows,center=C)
