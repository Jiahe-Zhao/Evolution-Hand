"""Long-side grasp validation: preserve planned contact coordinates during closure."""
import os,sys,json,argparse,math,re,hashlib,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'contact_control_calibration_20261021'))
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old=fi._fast_walk
def walk(path,*a,**k):
    if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
    yield from old(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser();p.add_argument('--source',default='/home/zjh/Evolution_PC/outputs/motion_collision_revision_20261021/matrix/baseline/right/urdf/current_agent.urdf');p.add_argument('--plans',default=str(BASE/'plans.json'));p.add_argument('--plan-index',type=int,default=0);p.add_argument('--output',required=True);p.add_argument('--record',action='store_true');p.add_argument('--dt',type=float,default=1/240);p.add_argument('--kp',type=float,default=6.);p.add_argument('--kd',type=float,default=.6);p.add_argument('--contact-k',type=float,default=3000.);p.add_argument('--contact-d',type=float,default=10.);p.add_argument('--torsion-radius',type=float,default=0.);p.add_argument('--mass',type=float,default=.08);p.add_argument('--offset-mm',type=float,default=0.);p.add_argument('--force-target',type=float,default=2.);p.add_argument('--duration',type=float,default=15.);p.add_argument('--hand-gravity',action='store_true');p.add_argument('--close-speed',type=float,default=.0006);p.add_argument('--no-lift',action='store_true');p.add_argument('--orientation-weight',type=float,default=.003);p.add_argument('--staged-force',action='store_true');p.add_argument('--bounded-ik',action='store_true');p.add_argument('--pad-face',default='x',choices=['x','yminus','yplus','ymixed']);p.add_argument('--freeze-after-grasp',action='store_true');p.add_argument('--adaptive-force',action='store_true');p.add_argument('--joint-pulse',type=float,default=0.)
p.add_argument('--offset-y-mm',type=float,default=0.);p.add_argument('--yaw-deg',type=float,default=0.)
p.add_argument('--num-envs',type=int,default=256);p.add_argument('--epochs',type=int,default=128);p.add_argument('--bc-steps',type=int,default=800);p.add_argument('--eval-steps',type=int,default=1200);p.add_argument('--smoke',action='store_true')
p.add_argument('--isolate',type=int,default=1);p.add_argument('--standard-reset',type=int,default=1);p.add_argument('--checkpoint',required=True);p.add_argument('--probe-steps',type=int,default=1800)
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
from isaacsim.core.utils.prims import create_prim
for i in range(args.num_envs):create_prim(f'/World/envs/env_{i}',translation=((i%16)*1.0,(i//16)*1.0,0.))
cfgpath=out/'hand_cfg.py';parse_urdf_and_generate_articulation_cfg(str(source),str(source),str(cfgpath))
spec=importlib.util.spec_from_file_location('calibration_cfg',cfgpath);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);cfg=mod.CURRENT_HAND_CFG.replace(prim_path='/World/envs/env_.*/Hand')
cfg.spawn.asset_path=str(path);cfg.spawn.usd_dir=str(out/'usd');cfg.spawn.force_usd_conversion=True;cfg.spawn.make_instanceable=False;cfg.spawn.merge_fixed_joints=False;cfg.spawn.rigid_props.disable_gravity=not args.hand_gravity
cfg.init_state.pos=tuple(root_position);cfg.init_state.rot=tuple(quat);cfg.init_state.joint_pos={re.escape(name):value for name,value in row['q'].items()};cfg.init_state.joint_pos.update({name:0. for name in NAMES});cfg.init_state.joint_vel={'.*':0.}
cfg.actuators['hand'].joint_names_expr=[re.escape(n) for n in model.names];cfg.actuators['hand'].stiffness=args.kp;cfg.actuators['hand'].damping=args.kd
cfg.actuators['wrist']=ImplicitActuatorCfg(joint_names_expr=NAMES,effort_limit_sim=100.,velocity_limit_sim=.02,stiffness=3000.,damping=100.,armature=0.)
hand=Articulation(cfg)
from fixed_clusters import apply as apply_fixed_filters
fixed_pairs=[apply_fixed_filters(ctx.stage,path,f'/World/envs/env_{i}/Hand') for i in range(args.num_envs)]
obj=RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/Object',spawn=sim.CuboidCfg(size=(.018,.018,.18),activate_contact_sensors=True,mass_props=sim.MassPropertiesCfg(mass=args.mass),rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45,.24,.09))),init_state=RigidObjectCfg.InitialStateCfg(pos=(args.offset_mm/1000,args.offset_y_mm/1000,.3),rot=(math.cos(math.radians(args.yaw_deg)/2),0.,0.,math.sin(math.radians(args.yaw_deg)/2)))))
support=RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/Support',spawn=sim.CuboidCfg(size=(.03,.03,.008),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.2,.28,.4))),init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,.206))))
plate=RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/Plate',spawn=sim.CuboidCfg(size=(.026,.035,.012),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.24,.32,.45))),init_state=RigidObjectCfg.InitialStateCfg(pos=(.05,0,.219))))
materials=configure(ctx.stage,'/World')
for prim in ctx.stage.Traverse():
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
# The scene above is generated from the validated wrapping test, with only batched paths.
import gym,yaml,time
from rl_games.common import env_configurations,vecenv
from rl_games.torch_runner import Runner
if args.isolate:
 from isaacsim.core.cloner import GridCloner
 physics_path=next(str(p.GetPath()) for p in ctx.stage.Traverse() if p.HasAPI(PhysxSchema.PhysxSceneAPI))
 GridCloner(spacing=1.).filter_collisions(physics_path,'/World/collisions',[f'/World/envs/env_{i}' for i in range(args.num_envs)],global_paths=[])
ctx.reset();hand.write_joint_state_to_sim(hand.data.default_joint_pos,torch.zeros_like(hand.data.default_joint_pos));hand.set_joint_position_target(hand.data.default_joint_pos);ctx.forward();hand.update(args.dt);obj.update(args.dt)
dev=hand.device;N=args.num_envs;dt=1/60;decimation=4
assert abs(args.dt-1/240)<1e-10
limits=hand.root_physx_view.get_dof_limits().to(dev)[0]
wids=[hand.joint_names.index(n) for n in NAMES];fids=[i for i in range(len(hand.joint_names)) if i not in wids]
assert len(fids)==21
root_id=hand.body_names.index('link_0_0');root0=hand.data.body_pos_w[:,root_id].clone();origins=obj.data.root_pos_w.clone()-torch.tensor([0,0,.3],device=dev)
assert (root0-origins-torch.tensor(root_position,device=dev)).abs().max()<1e-4
initial_object=obj.data.root_state_w.clone();initial_q=hand.data.default_joint_pos.clone();rootq=torch.tensor(quat,dtype=torch.float32,device=dev).expand(N,-1)
import omni.physics.tensors as tensors
sensor=tensors.create_simulation_view('torch');sensor.set_subspace_roots('/')
body_views=[]
for digit in range(1,6):
 body_views.append([sensor.create_rigid_contact_view('/World/envs/env_*/Hand/'+name,filter_patterns=['/World/envs/env_*/Object']) for name in hand.body_names if name.startswith(f'link_{digit}_')])
plane_view=sensor.create_rigid_contact_view('/World/envs/env_*/Object',filter_patterns=['/World/envs/env_*/Plate','/World/envs/env_*/Support'])
def forces():
 return torch.stack([sum((v.get_contact_force_matrix(dt=args.dt).reshape(N,-1,3).sum(1) for v in group),torch.zeros((N,3),device=dev)) for group in body_views],dim=1)
source_config=Path('/home/zjh/Evolution_PC/outputs/strike_v3_training_20261020/bc_seeds/strike/ppo/params/agent.yaml')
config=yaml.safe_load(source_config.read_text());original_config=json.loads(json.dumps(config));pc=config['params']['config']
pc.update(num_actors=N,max_epochs=args.epochs,env_name='wrap_probe',train_dir=str(out),full_experiment_name='ppo',save_frequency=32,save_best_after=0)
if args.smoke:pc['minibatch_size']=N*pc['horizon_length']
assert N*pc['horizon_length']%pc['minibatch_size']==0
config['params']['load_checkpoint']=False;config['params']['load_path']=''
(out/'effective_agent.yaml').write_text(yaml.safe_dump(config))
# No geometric jitter is introduced here: initial scene is the audited nominal pose.
# Independent positional robustness is measured by perturb_grasp.py.
class WrapEnv(vecenv.IVecEnv):
 def __init__(self):
  self.num_envs=N;self.action_space=gym.spaces.Box(-1,1,(24,),dtype=np.float32);self.observation_space=gym.spaces.Box(-5,5,(67,),dtype=np.float32)
  self.target=initial_q.clone();self.wrist=torch.zeros((N,3),device=dev);self.age=torch.zeros(N,device=dev,dtype=torch.long);self.contact=torch.zeros((N,5,3),device=dev)
  self.hold=torch.zeros(N,device=dev);self.best=torch.zeros(N,device=dev);self.peak_rise=torch.zeros(N,device=dev);self.total_episodes=0;self.success_episodes=0;self.falls=0;self.max_limit=0.;self.max_speed=0.;self.max_force=0.;self.speed_bursts=0;self.nonfinite=0
 def get_env_info(self):return {'observation_space':self.observation_space,'action_space':self.action_space,'agents':1,'value_size':1}
 def get_number_of_agents(self):return 1
 def set_train_info(self,*a,**k):pass
 def get_env_state(self):return None
 def set_env_state(self,state):pass
 def obs(self):
  q=2*(hand.data.joint_pos[:,fids]-limits[fids,0])/(limits[fids,1]-limits[fids,0])-1
  tar=2*(self.target[:,fids]-limits[fids,0])/(limits[fids,1]-limits[fids,0])-1
  objstate=obj.data.root_state_w.clone();objstate[:,:3]=(objstate[:,:3]-origins-torch.tensor([0,0,.3],device=dev))/.1;objstate[:,10:13]/=10
  return torch.cat([q,tar,(hand.data.body_pos_w[:,root_id]-root0)/.1,self.wrist/.1,objstate,self.contact[:,:,0].abs()/2,(self.age.float()*dt/50)[:,None]],1).clamp(-5,5)
 def reset_ids(self,ids):
  if not len(ids):return
  hand.write_joint_state_to_sim(initial_q[ids],torch.zeros_like(initial_q[ids]),env_ids=ids);obj.write_root_state_to_sim(initial_object[ids],env_ids=ids)
  if args.standard_reset:hand.reset(ids);obj.reset(ids)
  self.target[ids]=initial_q[ids];self.wrist[ids]=0;self.age[ids]=0;self.contact[ids]=0;self.hold[ids]=0;self.best[ids]=0;self.peak_rise[ids]=0
  hand.set_joint_position_target(self.target);hand.write_data_to_sim();obj.write_data_to_sim();ctx.forward();hand.update(args.dt);obj.update(args.dt)
 def reset(self):
  self.reset_ids(torch.arange(N,device=dev));return self.obs()
 def step(self,actions):
  oldstate=obj.data.root_state_w.clone();oldq=hand.data.joint_pos.clone()
  actions=actions.to(dev).clamp(-1,1)
  self.target[:,fids]=(self.target[:,fids]+actions[:,:21]*.25*dt).clamp(limits[fids,0],limits[fids,1])
  self.wrist+=actions[:,21:]*.02*dt
  local_wrist=quat_apply_inverse(rootq,self.wrist).clamp(limits[wids,0],limits[wids,1]);self.wrist=quat_apply(rootq,local_wrist);self.target[:,wids]=local_wrist
  hand.set_joint_position_target(self.target)
  for _ in range(decimation):
   hand.write_data_to_sim();obj.write_data_to_sim();ctx.step(render=False);hand.update(args.dt);obj.update(args.dt)
  self.age+=1;self.contact=forces();rise=obj.data.root_pos_w[:,2]-initial_object[:,2]
  support_force=plane_view.get_contact_force_matrix(dt=args.dt).reshape(N,-1,3).norm(dim=-1).amax(1)
  normal=self.contact[:,:,0].abs();touch=(normal[:,0]>.1)&(normal[:,1:].sum(1)>.1)
  upright=quat_apply(obj.data.root_quat_w,torch.tensor([0.,0.,1.],device=dev).expand(N,-1))[:,2]
  held=(rise>.015)&(support_force<.05)&touch&(upright>.98)
  self.hold=torch.where(held,self.hold+dt,torch.zeros_like(self.hold));self.best=torch.maximum(self.best,self.hold);self.peak_rise=torch.maximum(self.peak_rise,rise)
  # Dense grasp/lift diagnostic reward; not the production Strike reward.
  reward=normal[:,0].clamp(0,2)/2+normal[:,1:].clamp(0,.5).sum(1)/2+10*(rise/.03).clamp(0,1)+5*held.float()-.01*actions.square().mean(1)
  finite=torch.isfinite(hand.data.joint_pos).all(1)&torch.isfinite(obj.data.root_state_w).all(1)
  if not finite.all():self.nonfinite+=int((~finite).sum());raise RuntimeError('Nonfinite physical state')
  violation=torch.maximum(limits[:,0]-hand.data.joint_pos,hand.data.joint_pos-limits[:,1]).clamp_min(0).max()
  speed=obj.data.root_lin_vel_w.norm(dim=-1);self.max_limit=max(self.max_limit,float(violation));self.max_speed=max(self.max_speed,float(speed.max()));self.max_force=max(self.max_force,float(self.contact.norm(dim=-1).max()));self.speed_bursts+=int(((speed>2)&(obj.data.root_pos_w[:,2]>.2)&(self.age>6)).sum())
  suspicious=((speed>2)&(obj.data.root_pos_w[:,2]>.2))|(self.contact.norm(dim=-1).amax(1)>25)
  if suspicious.any() and len(events)<12:
   i=int(suspicious.nonzero()[0]);event=dict(step=step,env=i,age=int(self.age[i]),old_object=oldstate[i].cpu().tolist(),object=obj.data.root_state_w[i].cpu().tolist(),origin=origins[i].cpu().tolist(),old_q=oldq[i].cpu().tolist(),q=hand.data.joint_pos[i].cpu().tolist(),qdot=hand.data.joint_vel[i].cpu().tolist(),target=self.target[i].cpu().tolist(),digit_forces=self.contact[i].cpu().tolist(),actions=actions[i].cpu().tolist());events.append(event)
   print('ANOMALY',step,i,event['age'],float(speed[i]),float(self.contact[i].norm(dim=-1).max()),flush=True)
  fell=obj.data.root_pos_w[:,2]<.15;timeout=self.age>=1200;done=fell|timeout
  ids=done.nonzero().flatten();self.total_episodes+=len(ids);self.falls+=int(fell.sum());self.success_episodes+=int((self.best[ids]>=3).sum())
  info={'time_outs':timeout,'episode':{}}
  self.reset_ids(ids)
  return self.obs(),reward,done,info
 def diagnostics(self):return dict(episodes=self.total_episodes,successful_episodes=self.success_episodes,falls=self.falls,nonfinite=self.nonfinite,max_joint_limit_excess_rad=self.max_limit,max_object_speed_m_s=self.max_speed,max_digit_force_N=self.max_force,speed_bursts_above_workspace=self.speed_bursts)
env=WrapEnv();env_configurations.register('wrap_probe',{'vecenv_type':'WrapProbe','env_creator':lambda **kwargs:env});vecenv.register('WrapProbe',lambda config_name,num_actors,**kwargs:env)
torch.manual_seed(42);runner=Runner();runner.load(config);runner.reset();agent=runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)

agent.restore(args.checkpoint);agent.model.eval();torch.manual_seed(123);events=[];ob=env.reset()
for step in range(args.probe_steps):
 with torch.no_grad():result=agent.model({'is_train':False,'prev_actions':None,'obs':ob,'rnn_states':None})
 ob,reward,done,info=env.step(result['actions'])
 if step%300==0:print('PROBE',step,env.diagnostics(),flush=True)
report=dict(status='completed',steps=args.probe_steps,num_envs=N,isolation=bool(args.isolate),standard_reset=bool(args.standard_reset),checkpoint=args.checkpoint,diagnostics=env.diagnostics(),events=events)
(out/'diagnostic_report.json').write_text(json.dumps(report,indent=2));print('DIAGNOSTIC',json.dumps({k:v for k,v in report.items() if k!='events'}),flush=True);os._exit(0)
