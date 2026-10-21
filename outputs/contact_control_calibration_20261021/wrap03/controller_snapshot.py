"""Long-side grasp validation: preserve planned contact coordinates during closure."""
import os,sys,json,argparse,math,re,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old=fi._fast_walk
def walk(path,*a,**k):
    if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
    yield from old(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser();p.add_argument('--source',default='/home/zjh/Evolution_PC/outputs/motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf');p.add_argument('--plans',default=str(BASE/'plans.json'));p.add_argument('--plan-index',type=int,default=0);p.add_argument('--output',required=True);p.add_argument('--record',action='store_true');p.add_argument('--dt',type=float,default=1/240);p.add_argument('--kp',type=float,default=6.);p.add_argument('--kd',type=float,default=.6);p.add_argument('--contact-k',type=float,default=3000.);p.add_argument('--contact-d',type=float,default=10.);p.add_argument('--torsion-radius',type=float,default=0.);p.add_argument('--mass',type=float,default=.08);p.add_argument('--offset-mm',type=float,default=0.);p.add_argument('--force-target',type=float,default=2.);p.add_argument('--duration',type=float,default=15.);p.add_argument('--hand-gravity',action='store_true');p.add_argument('--close-speed',type=float,default=.0006);p.add_argument('--no-lift',action='store_true');p.add_argument('--orientation-weight',type=float,default=.003);p.add_argument('--staged-force',action='store_true');p.add_argument('--bounded-ik',action='store_true');p.add_argument('--pad-face',default='x',choices=['x','yminus','yplus','ymixed']);p.add_argument('--freeze-after-grasp',action='store_true');p.add_argument('--adaptive-force',action='store_true');p.add_argument('--joint-pulse',type=float,default=0.)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=args.enable_cameras or args.record
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'controller_snapshot.py')
from diagnostic_wrist import build,NAMES
source=Path(args.source);path,contract=build(source,out/'wrist')
app=AppLauncher(args).app
import torch,numpy as np,importlib.util
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear
import isaaclab.sim as sim
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
from mechanical_contact_materials import configure
from plan_grasp import Model
from pxr import Usd,UsdShade,UsdPhysics,PhysxSchema
model=Model(source,face=args.pad_face);plans=json.loads(Path(args.plans).read_text());feasible=[r for r in plans if not r['intersections']]
if not feasible:raise RuntimeError('No nonpenetrating initial plan')
row=feasible[args.plan_index];G=np.array(row['rotation']).T;center=np.array([0.,0.,.3]);root_position=center+G@(model.root[:3,3]-np.array(row['center']));root_R=G@model.root[:3,:3];quat=Rotation.from_matrix(root_R).as_quat()[[3,0,1,2]]
ctx=sim.SimulationContext(sim.SimulationCfg(dt=args.dt,device=args.device,gravity=(0,0,-9.81)))
cfgpath=out/'hand_cfg.py';parse_urdf_and_generate_articulation_cfg(str(source),str(source),str(cfgpath))
spec=importlib.util.spec_from_file_location('calibration_cfg',cfgpath);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);cfg=mod.CURRENT_HAND_CFG.replace(prim_path='/World/Hand')
cfg.spawn.asset_path=str(path);cfg.spawn.usd_dir=str(out/'usd');cfg.spawn.force_usd_conversion=True;cfg.spawn.make_instanceable=False;cfg.spawn.merge_fixed_joints=False;cfg.spawn.rigid_props.disable_gravity=not args.hand_gravity
cfg.init_state.pos=tuple(root_position);cfg.init_state.rot=tuple(quat);cfg.init_state.joint_pos={re.escape(name):value for name,value in row['q'].items()};cfg.init_state.joint_pos.update({name:0. for name in NAMES});cfg.init_state.joint_vel={'.*':0.}
cfg.actuators['hand'].joint_names_expr=[re.escape(n) for n in model.names];cfg.actuators['hand'].stiffness=args.kp;cfg.actuators['hand'].damping=args.kd
cfg.actuators['wrist']=ImplicitActuatorCfg(joint_names_expr=NAMES,effort_limit_sim=100.,velocity_limit_sim=.02,stiffness=3000.,damping=100.,armature=0.)
hand=Articulation(cfg)
from fixed_clusters import apply as apply_fixed_filters
fixed_pairs=apply_fixed_filters(ctx.stage,path,"/World/Hand")
obj=RigidObject(RigidObjectCfg(prim_path='/World/Object',spawn=sim.CuboidCfg(size=(.018,.018,.18),activate_contact_sensors=True,mass_props=sim.MassPropertiesCfg(mass=args.mass),rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45,.24,.09))),init_state=RigidObjectCfg.InitialStateCfg(pos=(args.offset_mm/1000,0,.3))))
support=RigidObject(RigidObjectCfg(prim_path='/World/Support',spawn=sim.CuboidCfg(size=(.03,.03,.008),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.2,.28,.4))),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,.206))))
plate=RigidObject(RigidObjectCfg(prim_path='/World/Plate',spawn=sim.CuboidCfg(size=(.026,.035,.012),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.24,.32,.45))),init_state=RigidObjectCfg.InitialStateCfg(pos=(.05,0,.219))))
materials=configure(ctx.stage,'/World')
for prim in Usd.PrimRange(ctx.stage.GetPrimAtPath('/World/Hand')):
    if prim.GetTypeName()=='Mesh' and 'finger_pad' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
        phys=PhysxSchema.PhysxCollisionAPI.Apply(prim);phys.CreateTorsionalPatchRadiusAttr(args.torsion_radius)
for prim in ctx.stage.Traverse():
    if prim.HasAPI(PhysxSchema.PhysxMaterialAPI) and prim.GetName()=='pad':
        mat=PhysxSchema.PhysxMaterialAPI(prim);mat.CreateCompliantContactStiffnessAttr(args.contact_k);mat.CreateCompliantContactDampingAttr(args.contact_d)
# Re-read effective values after experiment overrides; configure() returned pre-override values.
for material_row in materials:
    primitive=ctx.stage.GetPrimAtPath(material_row['path'])
    bound=UsdShade.MaterialBindingAPI(primitive).ComputeBoundMaterial('physics')[0]
    material_api=PhysxSchema.PhysxMaterialAPI(bound.GetPrim())
    material_row['compliant_stiffness']=material_api.GetCompliantContactStiffnessAttr().Get()
    material_row['compliant_damping']=material_api.GetCompliantContactDampingAttr().Get()
    if material_row['role']=='pad':
        assert abs(material_row['compliant_stiffness']-args.contact_k)<1e-6
        assert abs(material_row['compliant_damping']-args.contact_d)<1e-6
camera=writer=None
if args.record:
    import imageio.v2 as imageio
    from isaaclab.sensors import Camera,CameraCfg
    light=sim.DomeLightCfg(intensity=850.,color=(.72,.78,.9),visible_in_primary_ray=False);light.func('/World/Light',light)
    floor=sim.CuboidCfg(size=(2.,2.,.005),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.14,.18,.23),roughness=.9));floor.func('/World/Ground',floor,translation=(0,0,.05))
    appearance={}
    for role,color in {'pad':(.08,.55,.32),'bearing':(.65,.36,.10),'body':(.28,.35,.42)}.items():
        mat=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.55);mat.func('/World/Appearance/'+role,mat);appearance[role]=UsdShade.Material(ctx.stage.GetPrimAtPath('/World/Appearance/'+role))
    for prim in Usd.PrimRange(ctx.stage.GetPrimAtPath('/World/Hand')):
        if prim.GetTypeName()=='Mesh':
            name=str(prim.GetPath());role='pad' if 'finger_pad' in name else ('bearing' if 'bearing' in name or 'shaft' in name else 'body');UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance[role],bindingStrength='strongerThanDescendants')
    camera=Camera(CameraCfg(prim_path='/World/Camera',height=608,width=800,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=26.,clipping_range=(.001,10.))))
    writer=imageio.get_writer(str(out/'validation.mp4'),fps=24,codec='libx264',quality=7)
ctx.reset();hand.write_joint_state_to_sim(hand.data.default_joint_pos,torch.zeros_like(hand.data.default_joint_pos));hand.set_joint_position_target(hand.data.default_joint_pos);ctx.forward();hand.update(args.dt);obj.update(args.dt)
if camera:
    target=torch.tensor([[0.,0.,.35]],device=args.device);camera.set_world_poses_from_view(target+torch.tensor([[.35,-.40,.25]],device=args.device),target)
    for _ in range(5):ctx.render()
import omni.physics.tensors as tensors
sensor=tensors.create_simulation_view('torch');sensor.set_subspace_roots('/')
pad_bodies=[];locals_=[];normals=[];finger_ids=[]
for digit,name,point,normal in model.pad_specs:
    if name not in hand.body_names:raise RuntimeError('Fixed pad body was unexpectedly merged: '+name)
    pad_bodies.append(hand.body_names.index(name));locals_.append(point);normals.append(normal)
    finger_ids.append([hand.joint_names.index(n) for n in model.names if n.startswith(f'link_{digit}_') or f'to_link_{digit}_' in n])
pad_views=[sensor.create_rigid_contact_view('/World/Hand/'+hand.body_names[b],filter_patterns=['/World/Object']) for b in pad_bodies]
# A wrapping grasp can carry load through middle phalanges as well as the tips.
views=[sensor.create_rigid_contact_view(f'/World/Hand/link_{digit}_*',filter_patterns=['/World/Object']) for digit in range(1,6)]
middle_views=[sensor.create_rigid_contact_view(f'/World/Hand/link_{digit}_1',filter_patterns=['/World/Object']) for digit in range(1,6)]
object_view=sensor.create_rigid_contact_view('/World/Object',filter_patterns=['/World/Plate','/World/Support'])
dev=hand.device;offsets=torch.tensor(np.array(locals_),dtype=torch.float32,device=dev);normal_local=torch.tensor(np.array(normals),dtype=torch.float32,device=dev)
limits=hand.root_physx_view.get_dof_limits().to(dev)[0];wids=[hand.joint_names.index(n) for n in NAMES];qtarget=hand.data.default_joint_pos.clone();base_q=qtarget.clone();wzero=torch.zeros((1,3),device=dev);rootq=torch.tensor(quat,dtype=torch.float32,device=dev)[None];root_id=hand.body_names.index('link_0_0');root0=hand.data.body_pos_w[:,root_id].clone()
def pads():
    r=hand.data.body_quat_w[:,pad_bodies];return hand.data.body_pos_w[:,pad_bodies]+quat_apply(r,offsets[None]),quat_apply(r,normal_local[None])
pp,nn=pads();local=pp-torch.tensor(center,dtype=torch.float32,device=dev)[None,None,:]
# The planner centers the thumb within its reachable workspace. Keep that point
# during closure instead of adding an unplanned lateral/axial thumb motion.
sign=torch.tensor([[1.,-1.,-1.,-1.,-1.]],device=dev);normal_desired=torch.zeros((1,5,3),device=dev);normal_desired[:,:,0]=-sign;normal_reference=nn.clone()
depth=torch.zeros((1,5),device=dev);forces_goal=torch.tensor([[args.force_target]+[args.force_target/4]*4],device=dev)
filtered_force=torch.zeros((1,5),device=dev);touch_stable=0.;touch_time=None;grasp_obj=None;grasp_shift=None;grasp_quat=None;contact_mask=None;held_q=None;tap_hold_z=None
initial_obj=obj.data.root_state_w.clone();trace=[];stable=0.;lift_time=None;reason='duration';peak_limit=0.;free_seconds=0.;phase='settle';dtcontrol=1/60;decimation=round(dtcontrol/args.dt)
audit={'args':vars(args),'fixed_cluster_filter_pairs':fixed_pairs,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'initial_pad_positions':pp[0].cpu().tolist(),'initial_pad_normals':nn[0].cpu().tolist(),'initial_q':hand.data.joint_pos[0].cpu().tolist(),'plan':row,'wrist_contract':contract,'joint_names':hand.joint_names,'body_names':hand.body_names,'materials':materials,'force_targets_N':forces_goal.cpu().tolist(),'object_free':True,'object_mass_kg':args.mass,'object_size_m':[.018,.018,.18],'hand_gravity':args.hand_gravity,'jacobian_shape':list(hand.root_physx_view.get_jacobians().shape),'runtime_materials':hand.root_physx_view.get_material_properties().cpu().tolist(),'runtime_kp':hand.root_physx_view.get_dof_stiffnesses().cpu().tolist(),'runtime_kd':hand.root_physx_view.get_dof_dampings().cpu().tolist()}
jacobian_errors=[]
initial_jac=hand.root_physx_view.get_jacobians()
qfk=np.array([row['q'][n] for n in model.names])
for f,(body,ids) in enumerate(zip(pad_bodies,finger_ids)):
    J=initial_jac[:,body-1,:,ids];lever=quat_apply(hand.data.body_quat_w[:,body],offsets[f][None]-hand.data.body_com_pos_b[:,body]);jp=J[:,:3,:]+torch.cross(J[:,3:,:].transpose(1,2),lever[:,None,:].expand(-1,len(ids),-1),dim=-1).transpose(1,2)
    finite=[]
    for jid in ids:
        k=model.names.index(hand.joint_names[jid]);plus=qfk.copy();minus=qfk.copy();plus[k]+=1e-5;minus[k]-=1e-5
        finite.append(G@((model.pads(plus)[0][f]-model.pads(minus)[0][f])/2e-5))
    jacobian_errors.append(float(np.max(np.abs(jp[0].cpu().numpy()-np.array(finite).T))))
audit['point_jacobian_fd_error_m_per_rad']=jacobian_errors
(out/'audit.json').write_text(json.dumps(audit,indent=2))
print('JACOBIAN_FD_ERRORS',jacobian_errors,flush=True)
if max(jacobian_errors)>.005:raise RuntimeError('Point Jacobian mismatch; do not trust grasp controller')
def smooth(u):
    u=max(0.,min(1.,u));return u**3*(10-15*u+6*u*u)
for step in range(round(args.duration/args.dt)):
    t=step*args.dt
    if step%decimation==0:
        pp,nn=pads();vectors=torch.stack([v.get_contact_force_matrix(dt=args.dt).reshape(-1,3).sum(0) for v in views])[None]
        normal_force=abs((vectors*normal_desired).sum(-1));force_norm=torch.linalg.vector_norm(vectors,dim=-1)
        filtered_force=.7*filtered_force+.3*normal_force
        if bool(filtered_force[0,0]>.035) and int((filtered_force[0,1:]>.035).sum())>=2:touch_stable+=dtcontrol
        else:touch_stable=max(0.,touch_stable-dtcontrol)
        if touch_time is None and touch_stable>.25:
            touch_time=t;contact_mask=(filtered_force[0,1:]>.035)
            if args.adaptive_force:
                forces_goal[0,1:]=.12;forces_goal[0,1:][contact_mask]=args.force_target/int(contact_mask.sum())
        active_goal=forces_goal if not args.staged_force else (torch.ones_like(forces_goal)*.12 if touch_time is None else torch.ones_like(forces_goal)*.12+(forces_goal-.12)*smooth((t-touch_time)/1.5))
        actual_shift=hand.data.body_pos_w[:,root_id]-root0
        if t>.75 and not (args.freeze_after_grasp and lift_time is not None):
            phase='close' if lift_time is None else phase
            depth+=((active_goal-filtered_force)/active_goal.clamp_min(.1)).clamp(-1,1)*args.close_speed*dtcontrol
            depth.clamp_(-.004,.010)
            targets=torch.tensor(center,dtype=torch.float32,device=dev)[None,None,:]+actual_shift[:,None,:]+local
            targets[:,:,0]=torch.tensor(center[0],device=dev)+actual_shift[:,0,None]+local[:,:,0]-sign*depth
            Jall=hand.root_physx_view.get_jacobians();desired=qtarget.clone()
            for f,(body,ids) in enumerate(zip(pad_bodies,finger_ids)):
                J=Jall[:,body-1,:,ids];lever=quat_apply(hand.data.body_quat_w[:,body],offsets[f][None]-hand.data.body_com_pos_b[:,body]);jp=J[:,:3,:]+torch.cross(J[:,3:,:].transpose(1,2),lever[:,None,:].expand(-1,len(ids),-1),dim=-1).transpose(1,2)
                error=targets[:,f]-pp[:,f];angle=torch.cross(nn[:,f],normal_reference[:,f],dim=-1)
                j=torch.cat([jp,J[:,3:,:]*args.orientation_weight],dim=1);rhs=torch.cat([error,angle*args.orientation_weight],dim=1);eye=torch.eye(len(ids),device=dev)[None]
                dq=torch.linalg.solve(j.transpose(1,2)@j+eye*1e-6,j.transpose(1,2)@rhs[:,:,None]).squeeze(-1)
                if args.bounded_ik:
                    qa=hand.data.joint_pos[0,ids].cpu().numpy();lim=limits[ids].cpu().numpy();lower=np.maximum(lim[:,0]-qa,-.12);upper=np.minimum(lim[:,1]-qa,.12);lower=np.minimum(lower,upper-1e-8)
                    aa=np.vstack([j[0].cpu().numpy(),np.eye(len(ids))*.001]);bb=np.r_[rhs[0].cpu().numpy(),np.zeros(len(ids))]
                    dq=torch.tensor(lsq_linear(aa,bb,bounds=(lower,upper),tol=1e-5,max_iter=30).x,dtype=torch.float32,device=dev)[None]
                desired[:,ids]=hand.data.joint_pos[:,ids]+dq.clamp(-.12,.12)
            delta=(desired-qtarget).clamp(-.25*dtcontrol,.25*dtcontrol);qtarget+=delta;qtarget=torch.maximum(torch.minimum(qtarget,limits[:,1]),limits[:,0])
            both=bool(filtered_force[0,0]>(args.force_target*.75 if args.adaptive_force else .7)) and float(filtered_force[0,1:].sum())>(args.force_target*.65 if args.adaptive_force else .3) and int((filtered_force[0,1:]>.15).sum())>=2 and float(normal_force.max())<args.force_target*2 and float(quat_apply(obj.data.root_quat_w,torch.tensor([[0.,0.,1.]],device=dev))[0,2])>math.cos(.2)
            stable=stable+dtcontrol if both else max(0.,stable-dtcontrol)
            if lift_time is None and stable>.75 and not args.no_lift:
                lift_time=t;phase='lift';grasp_obj=obj.data.root_pos_w.clone();grasp_shift=actual_shift.clone();grasp_quat=obj.data.root_quat_w.clone();held_q=qtarget.clone()
        wrist=wzero.clone();external=torch.zeros((1,1,3),device=dev);torque=external.clone()
        if lift_time is not None:
            elapsed=t-lift_time;wrist[:,2]=.03*smooth(elapsed/3.)
            phase='lift' if elapsed<3 else 'hold'
            if 6<=elapsed<7:external[:,:,0]=.4;phase='lateral_force'
            if 8<=elapsed<9:torque[:,:,1]=.003;phase='torque'
            if 10<=elapsed<13:wrist[:,0]=.005*math.sin(2*math.pi*(elapsed-10)/3);phase='swing'
            if elapsed>=14:wrist[:,0]=.05*smooth((elapsed-14)/5.);phase='transport'
            if elapsed>=19:
                if tap_hold_z is None and float(object_view.get_contact_force_matrix(dt=args.dt).reshape(-1,3)[0].norm())>.3:tap_hold_z=float(actual_shift[0,2])
                wrist[:,2]=tap_hold_z if tap_hold_z is not None else .03-.02*smooth((elapsed-19)/3.);phase='light_tap'
            if elapsed>=22.5:
                start_z=.01 if tap_hold_z is None else tap_hold_z
                wrist[:,2]=start_z+(.03-start_z)*smooth((elapsed-22.5)/2.5);phase='retract'
        if args.freeze_after_grasp and held_q is not None:
            qtarget=held_q.clone()
            if args.joint_pulse and 10<=t-lift_time<13:
                pulse=args.joint_pulse*math.sin(2*math.pi*(t-lift_time-10)/3)
                for jid,name in enumerate(hand.joint_names):
                    if jid not in wids and 'spread' not in name:qtarget[:,jid]+=pulse
                qtarget=torch.maximum(torch.minimum(qtarget,limits[:,1]),limits[:,0])
        qtarget[:,wids]=quat_apply_inverse(rootq,wrist)
        hand.set_joint_position_target(qtarget)
        obj.set_external_force_and_torque(quat_apply_inverse(obj.data.root_quat_w[:,None,:],external),quat_apply_inverse(obj.data.root_quat_w[:,None,:],torque))
        relative=obj.data.root_pos_w-initial_obj[:,:3]-actual_shift if grasp_obj is None else obj.data.root_pos_w-grasp_obj-(actual_shift-grasp_shift)
        angle=2*torch.acos(abs(obj.data.root_quat_w[:,0]).clamp(0,1)) if grasp_quat is None else 2*torch.acos(abs((obj.data.root_quat_w*grasp_quat).sum(-1)).clamp(0,1))
        excess=torch.maximum(limits[:,0]-hand.data.joint_pos[0],hand.data.joint_pos[0]-limits[:,1]).clamp_min(0);peak_limit=max(peak_limit,float(excess.max()))
        plane_vectors=object_view.get_contact_force_matrix(dt=args.dt).reshape(-1,3);plane_force=plane_vectors.sum(dim=0).cpu().tolist()
        rise=float(obj.data.root_pos_w[0,2]-initial_obj[0,2]);free_seconds=free_seconds+dtcontrol if rise>.015 and float(relative.norm())<.005 and float(angle[0])<.2 and float(plane_vectors.norm(dim=-1).max())<.05 else 0.
        trace.append({'t':t,'phase':phase,'q':hand.data.joint_pos[0].cpu().tolist(),'target':qtarget[0].cpu().tolist(),'pad_positions':pp[0].cpu().tolist(),'pad_normals':nn[0].cpu().tolist(),'pad_only_force_N':[float(v.get_contact_force_matrix(dt=args.dt).reshape(-1,3).sum(0).norm()) for v in pad_views],'middle_phalanx_force_N':[float(v.get_contact_force_matrix(dt=args.dt).reshape(-1,3).sum(0).norm()) for v in middle_views],'normal_force_N':normal_force[0].cpu().tolist(),'force_norm_N':force_norm[0].cpu().tolist(),'active_force_goal':active_goal[0].cpu().tolist(),'depth_m':depth[0].cpu().tolist(),'object':obj.data.root_state_w[0].cpu().tolist(),'wrist_actual':actual_shift[0].cpu().tolist(),'wrist_target':wrist[0].cpu().tolist(),'relative_slip_m':float(relative.norm()),'angle_rad':float(angle[0]),'rise_m':rise,'limit_excess_rad':float(excess.max()),'free_hold_seconds':free_seconds,'plane_force':plane_force,'plate_force_N':float(plane_vectors[0].norm()),'support_force_N':float(plane_vectors[1].norm()),'external_force_world':external[0,0].cpu().tolist(),'external_torque_world':torque[0,0].cpu().tolist()})
        if step%round(2/args.dt)==0:print('PROGRESS',round(t,2),phase,'N',normal_force[0].cpu().tolist(),'rise',rise,flush=True)
        if float(obj.data.root_pos_w[0,2])<.16 or float(obj.data.root_lin_vel_w.norm())>2.:reason='tool_lost_or_unstable';break
    hand.write_data_to_sim();obj.write_data_to_sim();ctx.step(render=False);hand.update(args.dt);obj.update(args.dt)
    if camera and step%max(1,round(1/(24*args.dt)))==0:
        from PIL import Image,ImageDraw
        ctx.render();camera.update(1/24,force_recompute=True);im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));d=ImageDraw.Draw(im);d.rectangle((0,0,800,53),fill=(18,25,38));d.text((12,8),f'FREE TOOL | LIVE ISAAC | {phase} | {t:.2f}s',fill='white');d.text((12,29),f'Normal forces: '+', '.join(f'{v:.2f}' for v in trace[-1]['normal_force_N'])+' N',fill='white');writer.append_data(np.asarray(im))
if writer:writer.close()
report={'reason':reason,'lift_time_s':lift_time,'max_free_hold_s':max(r['free_hold_seconds'] for r in trace),'max_lift_m':max(r['rise_m'] for r in trace),'max_limit_excess_rad':peak_limit,'peak_object_speed_m_s':max(np.linalg.norm(r['object'][7:10]) for r in trace),'last':trace[-1],'scope':'free object, physical wrist; no object state overwrite after reset','passed_static_hold':max(r['free_hold_seconds'] for r in trace)>=3.,'phase_coverage':sorted(set(r['phase'] for r in trace)), 'controller_sha256':hashlib.sha256((out/'controller_snapshot.py').read_bytes()).hexdigest(),'run_parameters':vars(args)}
(out/'trace.json').write_text(json.dumps(trace));(out/'report.json').write_text(json.dumps(report,indent=2));print('CALIBRATION_REPORT',json.dumps(report),flush=True);os._exit(0)
