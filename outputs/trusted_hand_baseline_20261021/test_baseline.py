"""Isolated import audit and shape-resolved pressure/holding bench."""
import os,sys,json,argparse,math
from pathlib import Path
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old=fi._fast_walk
def walk(path,*a,**k):
 if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
 yield from old(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['audit','press','hold'],required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.source=str(Path(args.source).resolve());args.output=str(Path(args.output).resolve());os.environ['EVOLUTION_ROOT']='/home/zjh/Evolution_PC';app=AppLauncher(args).app
import torch,numpy as np
import isaaclab.sim as sim
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import Usd,UsdPhysics,PhysxSchema,PhysicsSchemaTools
import omni.physx
from contact_materials import configure
from build_contact_bench import build
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
dt=1/240;ctx=sim.SimulationContext(sim.SimulationCfg(dt=dt,device=args.device,gravity=(0,0,-9.81)))
# IsaacLab suppresses CPU contact callbacks by default; enable only in this diagnostic.
ctx.carb_settings.set_bool('/physics/disableContactProcessing',False)
if args.mode=='audit':
 import importlib.util
 from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
 cfgpath=out/'hand_cfg.py';parse_urdf_and_generate_articulation_cfg(args.source,args.source,str(cfgpath))
 spec=importlib.util.spec_from_file_location('baseline_cfg',cfgpath);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);cfg=m.CURRENT_HAND_CFG.replace(prim_path='/World/Hand');cfg.init_state.pos=(0,0,.5)
else:
 path=build(args.source,out/'bench_model')
 cfg=ArticulationCfg(prim_path='/World/Hand',spawn=sim.UrdfFileCfg(asset_path=str(path),fix_base=True,merge_fixed_joints=False,joint_drive=sim.UrdfConverterCfg.JointDriveCfg(drive_type='force',target_type='position',gains=sim.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=10000.,damping=100.))),init_state=ArticulationCfg.InitialStateCfg(joint_pos={'.*':0.}),actuators={'jaws':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=10000.,damping=100.,effort_limit_sim=10.,velocity_limit_sim=.01)})
cfg.spawn.make_instanceable=False;cfg.spawn.usd_dir=str(out/'usd');cfg.spawn.force_usd_conversion=True
hand=Articulation(cfg)
obj=support=None
if args.mode!='audit':
 obj=RigidObject(RigidObjectCfg(prim_path='/World/Object',spawn=sim.CuboidCfg(size=(.018,.018,.04),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=args.mode=='press'),mass_props=sim.MassPropertiesCfg(mass=.08),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.3,.45,.7))),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,.15))))
 if args.mode=='hold':support=RigidObject(RigidObjectCfg(prim_path='/World/Support',spawn=sim.CuboidCfg(size=(.03,.03,.01),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg()),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,.125))))
rows=configure(ctx.stage,'/World',True)
for prim in Usd.PrimRange(ctx.stage.GetPrimAtPath('/World')):
 if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
current=[];callback_errors=[];raw_header_count=0
def contact(headers,data):
 global raw_header_count
 raw_header_count+=len(headers)
 try:
  for h in headers:
   a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
   if '/Object' not in a and '/Object' not in b:continue
   for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
    d=data[i];imp=np.array(d.impulse);force=float(np.linalg.norm(imp)/dt)
    current.append(dict(a=a,b=b,position=list(d.position),normal=list(d.normal),force_N=force))
 except Exception as exc:
  if not callback_errors:callback_errors.append(str(exc))
ctx.reset();hand.update(dt)
ctx.carb_settings.set_bool("/physics/disableContactProcessing",False)
subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
import omni.physics.tensors as tensors
sensor_sim=tensors.create_simulation_view('torch');sensor_sim.set_subspace_roots('/')
contact_views={side:sensor_sim.create_rigid_contact_view('/World/Hand/'+side,filter_patterns=['/World/Object']) for side in ('left','right')} if args.mode!='audit' else {}
audit=dict(device=args.device,source=args.source,mode=args.mode,dt=dt,gravity=[0,0,-9.81],materials=rows,body_names=hand.body_names,joint_names=hand.joint_names)
view=hand.root_physx_view
for name in ['get_masses','get_inertias','get_material_properties','get_dof_stiffnesses','get_dof_dampings','get_dof_armatures','get_dof_max_forces','get_dof_max_velocities','get_contact_offsets','get_rest_offsets']:
 if hasattr(view,name):
  try:audit[name]=getattr(view,name)().tolist()
  except Exception as exc:audit[name]={'error':str(exc)}
(out/'runtime_audit.json').write_text(json.dumps(audit,indent=2))
base=hand.data.default_joint_pos.clone();hand.write_joint_state_to_sim(base,torch.zeros_like(base));trace=[];first=None;released=False;release_step=None;anchor=None;held_seconds=0.;max_slip=0.;stop_reason='timeout';target=0.
steps=240 if args.mode=='audit' else 2400
for step in range(steps):
 current.clear()
 if args.mode=='audit':q=base
 else:
  # A physical, force-limited position drive closes at 1 mm/s.
  target=min(.0075,max(0,(step*dt-.5))*.001)
  if released:target=release_target
  q=base.clone();q[:,hand.joint_names.index('left_drive')]=target
  if args.mode=='hold':q[:,hand.joint_names.index('right_drive')]=target
 hand.set_joint_position_target(q);hand.write_data_to_sim();ctx.step();hand.update(dt)
 if obj is not None:obj.update(dt)
 if support is not None:support.update(dt)
 forces={'left':0.,'right':0.};pad_forces={'left':0.,'right':0.};events=[]
 for c in current:
  path=c['b'] if '/Object' in c['a'] else c['a'];side='left' if '/left/' in path else ('right' if '/right/' in path else None)
  if side:
   forces[side]+=c['force_N']
   if 'finger_pad' in path:pad_forces[side]+=c['force_N']
   events.append(c)
   if first is None and c['force_N']>.01:first=dict(step=step,t_s=(step+1)*dt,collider=path,force_N=c['force_N'])
 callback_forces=forces.copy()
 for side,cv in contact_views.items():
  forces[side]=float(torch.linalg.vector_norm(cv.get_contact_force_matrix(dt=dt).sum(dim=1)).item())
 state=obj.data.root_state_w[0].cpu().tolist() if obj is not None else None
 trace.append(dict(step=step,t_s=(step+1)*dt,target_m=target,joint_pos=hand.data.joint_pos[0].cpu().tolist(),force_N=forces,pad_force_N=pad_forces,object=state,contacts=events,callback_force_N=callback_forces,effort_N=hand.data.applied_torque[0].cpu().tolist()))
 if args.mode=='press' and forces['left']>=2.:
  stop_reason='2 N press reached';break
 if args.mode=='hold':
  if not released and min(forces.values())>=1.5:
   released=True;release_step=step;release_target=target;anchor=np.array(state[:3]);s=support.data.root_state_w.clone();s[:,2]-=.1;support.write_root_state_to_sim(s)
  elif released:
   slip=float(np.linalg.norm(np.array(state[:3])-anchor))
   max_slip=max(max_slip,slip)
   if slip>.003:stop_reason='slip exceeded 3 mm';break
   held_seconds=(step-release_step)*dt
   if held_seconds>=3.:stop_reason='3 s suspended hold completed';break
 if not bool(torch.isfinite(hand.data.joint_pos).all()):stop_reason='nonfinite state';break
report=dict(device=args.device,force_source='PhysX contact tensor; shape identity from CPU callbacks only',mode=args.mode,steps=step+1,first_contact=first,callback_errors=callback_errors,raw_contact_header_count=raw_header_count,stop_reason=stop_reason,released=released,release_step=release_step,held_seconds=held_seconds,final_force_N=forces,max_slip_m=max_slip,shape_reporting_available=raw_header_count>0,final_pad_force_N=pad_forces if raw_header_count else None,scope='isolated contact bench, not full-hand tool grasp or Strike success')
if args.mode=='audit':
 actual_materials=np.array(audit['get_material_properties'])[0]
 report['scope']='full-hand import and static parameter audit; not grasp success'
 report['pad_shape_count']=sum(r['role']=='pad' for r in rows)
 report['material_roles_verified']=bool(np.isclose(actual_materials[:,0],.8).any() and np.isclose(actual_materials[:,0],.2).any())
 report['passed']=bool(torch.isfinite(hand.data.joint_pos).all()) and len(hand.joint_names)==21 and report['pad_shape_count']==5 and report['material_roles_verified']
elif args.mode=='press':report['passed']=stop_reason=='2 N press reached' and not callback_errors
else:report['passed']=held_seconds>=3. and not callback_errors
(out/'trace.json').write_text(json.dumps(trace));(out/'report.json').write_text(json.dumps(report,indent=2));print('BASELINE_REPORT',json.dumps(report),flush=True);os._exit(0)
