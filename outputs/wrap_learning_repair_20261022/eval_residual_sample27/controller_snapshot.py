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
p.add_argument('--mode',choices=['bc','ppo','eval'],default='bc');p.add_argument('--checkpoint',default='');p.add_argument('--seed',type=int,default=42);p.add_argument('--push',action='store_true');p.add_argument('--dataset',default='');p.add_argument('--num-envs',type=int,default=256);p.add_argument('--epochs',type=int,default=128);p.add_argument('--bc-steps',type=int,default=8000);p.add_argument('--eval-steps',type=int,default=1200);p.add_argument('--smoke',action='store_true')
p.add_argument('--refit',action='store_true');p.add_argument('--recovery-augmentation',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--history',action='store_true');p.add_argument('--extra-data',default='');p.add_argument('--only-extra',action='store_true');p.add_argument('--teacher-mix',type=float,default=1.);p.add_argument('--target-skip',action='store_true');p.add_argument('--finger-skip',action='store_true');p.add_argument('--no-policy-time',action='store_true');p.add_argument('--retrieval-bc',action='store_true');p.add_argument('--retrieval-lookahead',type=int,default=0);p.add_argument('--obs-noise',type=float,default=.0005);p.add_argument('--bc-lr',type=float,default=3e-4);p.add_argument('--extra-fraction',type=float,default=-1.);p.add_argument('--diagnostic-ppo',action='store_true');p.add_argument('--finger-std',type=float,default=.0005);p.add_argument('--wrist-std',type=float,default=.0001);p.add_argument('--stable-kl',action='store_true',default=True);p.add_argument('--legacy-kl',action='store_false',dest='stable_kl');p.add_argument('--ppo-lr',type=float,default=1e-7);p.add_argument('--bc-anchor',type=float,default=0.);p.add_argument('--baseline-report',default='');p.add_argument('--no-target-input',action='store_true');p.add_argument('--contraction',type=float,default=0.);p.add_argument('--critic-warmup',type=int,default=0);p.add_argument('--critic-lr',type=float,default=3e-4);p.add_argument('--residual-bc',action='store_true');p.add_argument('--bc-prior',default='');p.add_argument('--sample-eval',action='store_true');p.add_argument('--action-seed',type=int,default=2027)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=args.enable_cameras or args.record
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'controller_snapshot.py')
(out/'execution.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv,args=vars(args),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
for helper in ('expert_feedback.py','retrieval_bc.py','ppo_numerics.py','bc_anchor.py','critic_warmup.py','residual_bc.py'):
 if (BASE/helper).exists():shutil.copyfile(BASE/helper,out/(Path(helper).stem+'_snapshot.py'))
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
pc['normalize_input']=False
pc['learning_rate']=args.ppo_lr
config['params']['network']['separate']=True
config['params']['load_checkpoint']=False;config['params']['load_path']=''
(out/'effective_agent.yaml').write_text(yaml.safe_dump(config))

# Fixed physical feature scaling: BC and PPO share exactly the same encoder.
# Position commands retain every original joint limit, with the original slew limits.
class WrapEnv(vecenv.IVecEnv):
 def __init__(self):
  self.num_envs=N
  self.action_space=gym.spaces.Box(-1,1,(24,),dtype=np.float32)
  self.observation_space=gym.spaces.Box(-5,5,(73 if args.history else 67,),dtype=np.float32)
  self.target=initial_q.clone();self.wrist=torch.zeros((N,3),device=dev)
  self.age=torch.zeros(N,device=dev,dtype=torch.long);self.contact=torch.zeros((N,5,3),device=dev)
  self.hold=torch.zeros(N,device=dev);self.best=torch.zeros(N,device=dev)
  self.filtered=torch.zeros((N,5),device=dev);self.stable=torch.zeros(N,device=dev)
  self.episode_falls=torch.zeros(N,device=dev,dtype=torch.long)
  self.max_speed=0.;self.max_force=0.;self.max_limit=0.;self.falls=0;self.episodes=0
  self.peak_lift=torch.zeros(N,device=dev);self.anomalies=[];self.reward_terms={}
  fixed=[(0,0,0),(-.5,0,0),(.5,0,0),(0,-.5,0),(0,.5,0),(0,0,-2),(0,0,2),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-.5,-.5,-1),(.5,.5,1)]
  rng=np.random.default_rng(args.seed)
  self.cases=np.array([fixed[i] if i<len(fixed) else (*rng.uniform(-.5,.5,2),rng.uniform(-1.5,1.5)) for i in range(N)])
  if args.collect:
   train_fixed=[fixed[i] for i in [0,1,5,7,8,9,12]]
   self.cases=np.array([train_fixed[i] if i<len(train_fixed) else (*rng.uniform(-.5,.5,2),rng.uniform(-1.5,1.5)) for i in range(N)])
 def get_env_info(self):return {'observation_space':self.observation_space,'action_space':self.action_space,'agents':1,'value_size':1}
 def get_number_of_agents(self):return 1
 def set_train_info(self,*a,**k):pass
 def get_env_state(self):return None
 def set_env_state(self,state):pass
 def obs(self):
  q=2*(hand.data.joint_pos[:,fids]-limits[fids,0])/(limits[fids,1]-limits[fids,0])-1
  tar=2*(self.target[:,fids]-limits[fids,0])/(limits[fids,1]-limits[fids,0])-1
  state=obj.data.root_state_w.clone();state[:,:3]=(state[:,:3]-origins-torch.tensor([0,0,.3],device=dev))/.05
  state[:,7:10]/=.1;state[:,10:13]/=2
  parts=[q,tar,(hand.data.body_pos_w[:,root_id]-root0)/.05,self.wrist/.05,state,self.contact[:,:,0].abs()/2,(self.age.float()*dt/25)[:,None]]
  if args.history:parts.extend([self.filtered/2,(self.stable/.75).clamp(0,1)[:,None]])
  return torch.cat(parts,1).clamp(-5,5)
 def reset_ids(self,ids):
  if not len(ids):return
  state=initial_object[ids].clone()
  perturb=torch.tensor(self.cases[ids.cpu().numpy()],device=dev,dtype=torch.float32)
  state[:,:2]+=perturb[:,:2]/1000
  yaw=perturb[:,2]*math.pi/180
  state[:,3]=torch.cos(yaw/2);state[:,4:6]=0;state[:,6]=torch.sin(yaw/2)
  hand.write_joint_state_to_sim(initial_q[ids],torch.zeros_like(initial_q[ids]),env_ids=ids)
  obj.write_root_state_to_sim(state,env_ids=ids);hand.reset(ids);obj.reset(ids)
  self.target[ids]=initial_q[ids];self.wrist[ids]=0;self.age[ids]=0;self.contact[ids]=0;self.hold[ids]=0
  self.filtered[ids]=0;self.stable[ids]=0
  hand.set_joint_position_target(self.target);hand.write_data_to_sim();obj.write_data_to_sim();ctx.forward();hand.update(args.dt);obj.update(args.dt)
 def reset(self):
  self.best.zero_();self.peak_lift.zero_();self.episode_falls.zero_();self.reset_ids(torch.arange(N,device=dev));return self.obs()
 def step(self,actions):
  actions=actions.to(dev).clamp(-1,1)
  desired=limits[fids,0]+(actions[:,:21]+1)*.5*(limits[fids,1]-limits[fids,0])
  self.target[:,fids]+=(desired-self.target[:,fids]).clamp(-.25*dt,.25*dt)
  self.wrist+=(actions[:,21:]*.25-self.wrist).clamp(-.02*dt,.02*dt)
  local=quat_apply_inverse(rootq,self.wrist).clamp(limits[wids,0],limits[wids,1]);self.wrist=quat_apply(rootq,local);self.target[:,wids]=local
  hand.set_joint_position_target(self.target)
  external=torch.zeros((N,1,3),device=dev);torque=torch.zeros_like(external)
  if args.push:
   pulse=(self.age>=1080)&(self.age<1110)
   idx=torch.arange(N,device=dev);external[:,0,0]=pulse*(idx%2*2-1)*.15
   torque[:,0,1]=pulse*(idx%3-1)*.0005
  obj.set_external_force_and_torque(quat_apply_inverse(obj.data.root_quat_w[:,None,:],external),quat_apply_inverse(obj.data.root_quat_w[:,None,:],torque))
  for _ in range(decimation):
   hand.write_data_to_sim();obj.write_data_to_sim();ctx.step(render=False);hand.update(args.dt);obj.update(args.dt)
  self.age+=1;self.contact=forces();rise=obj.data.root_pos_w[:,2]-initial_object[:,2]
  support_force=plane_view.get_contact_force_matrix(dt=args.dt).reshape(N,-1,3).norm(dim=-1).amax(1)
  normal=self.contact[:,:,0].abs();norm=self.contact.norm(dim=-1)
  self.filtered=.7*self.filtered+.3*normal
  ready=(self.filtered[:,0]>.7)&(self.filtered[:,1:].sum(1)>.3)&(norm[:,1:]>.1).all(1)&(normal.max(1).values<4)
  self.stable=torch.where(ready,self.stable+dt,(self.stable-dt).clamp_min(0))
  touch=(normal[:,0]>.1)&(normal[:,1:].sum(1)>.1)
  upright=quat_apply(obj.data.root_quat_w,torch.tensor([0.,0.,1.],device=dev).expand(N,-1))[:,2]
  speed=obj.data.root_lin_vel_w.norm(dim=-1)
  held=(rise>.015)&(support_force<.05)&touch&(upright>.98)&(speed<.1)
  self.hold=torch.where(held,self.hold+dt,torch.zeros_like(self.hold));self.best=torch.maximum(self.best,self.hold);self.peak_lift=torch.maximum(self.peak_lift,rise)
  # Height only pays while grasped. Excess force and velocity are soft costs,
  # never episode terminations or hidden constraints on allowable morphology.
  balance=torch.minimum(normal[:,0]/1.5,normal[:,1:].sum(1)/1.5).clamp(0,1)
  opposed=(self.contact[:,0,0]*self.contact[:,1:,0].sum(1)<0).float()
  quality=balance*opposed*torch.exp(-torch.relu(speed-.08)*10)*upright.clamp(0,1)
  overforce=torch.relu(norm-4).square().sum(1).clamp_max(100)
  overspeed=torch.relu(speed-.15).square().clamp_max(25)
  reward=balance+5*(rise/.03).clamp(0,1)*quality+3*held.float()-.1*overforce-2*overspeed
  self.reward_terms=dict(balance=float(balance.mean()),height=float(((rise/.03).clamp(0,1)*quality).mean()),overforce=float(overforce.mean()),overspeed=float(overspeed.mean()))
  if not torch.isfinite(hand.data.joint_pos).all() or not torch.isfinite(obj.data.root_state_w).all():raise RuntimeError('Nonfinite physical state')
  self.max_limit=max(self.max_limit,float(torch.maximum(limits[:,0]-hand.data.joint_pos,hand.data.joint_pos-limits[:,1]).clamp_min(0).max()))
  self.max_speed=max(self.max_speed,float(speed.max()));self.max_force=max(self.max_force,float(norm.max()))
  bad=(norm.max(1).values>25)|((speed>2)&(obj.data.root_pos_w[:,2]>.2))
  if bad.any() and len(self.anomalies)<20:
   i=int(bad.nonzero()[0]);self.anomalies.append(dict(env=i,age=int(self.age[i]),force_N=norm[i].cpu().tolist(),object=obj.data.root_state_w[i].cpu().tolist(),q=hand.data.joint_pos[i].cpu().tolist()))
  fell=obj.data.root_pos_w[:,2]<.15;timeout=self.age>=1800;done=fell|timeout
  self.falls+=int(fell.sum());self.episodes+=int(done.sum());self.episode_falls+=fell.long()
  self.reset_ids(done.nonzero().flatten())
  return self.obs(),reward,done,{'time_outs':timeout,'episode':{}}
 def diagnostics(self):return dict(episodes=self.episodes,falls=self.falls,max_speed_m_s=self.max_speed,max_digit_force_N=self.max_force,max_limit_rad=self.max_limit,anomalies=self.anomalies)

if args.stable_kl:
 from ppo_numerics import policy_kl,verify
 from rl_games.algos_torch import torch_ext
 torch_ext.policy_kl=policy_kl
 (out/'kl_verification.json').write_text(json.dumps(verify(),indent=2))
env=WrapEnv();env_configurations.register('wrap_probe',{'vecenv_type':'WrapProbe','env_creator':lambda **kwargs:env});vecenv.register('WrapProbe',lambda config_name,num_actors,**kwargs:env)
torch.manual_seed(args.seed);runner=Runner();runner.load(config);runner.reset();agent=runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
if hasattr(agent.scheduler,'max_lr'):agent.scheduler.max_lr=1e-5
if args.residual_bc:
 assert not (args.retrieval_bc or args.target_skip or args.finger_skip or args.no_target_input or args.no_policy_time)
 if args.bc_prior:agent.restore(args.bc_prior,set_epoch=False)
 from residual_bc import install
 install(agent.model.a2c_network)
if args.retrieval_bc:
 from retrieval_bc import install
 install(agent.model.a2c_network,torch.load(args.extra_data,map_location=dev,weights_only=False),args.retrieval_lookahead)
if args.no_target_input:
 def remove_target(module,inputs):
  feed=dict(inputs[0]);feed['obs']=feed['obs'].clone();feed['obs'][:,21:42]=0;feed['obs'][:,45:48]=0
  return (feed,)
 agent.model.a2c_network.register_forward_pre_hook(remove_target)
if args.no_policy_time:
 def remove_time(module,inputs):
  feed=dict(inputs[0]);feed['obs']=feed['obs'].clone();feed['obs'][:,66]=0
  return (feed,)
 agent.model.a2c_network.register_forward_pre_hook(remove_time)
if args.target_skip or args.finger_skip:
 def target_skip(module,inputs,output):
  mu,sigma,value,states=output;ob=inputs[0]['obs']
  previous=torch.cat([ob[:,21:42],ob[:,45:48]*.2],1)
  result=previous+.02*mu
  if args.finger_skip:result=torch.cat([result[:,:21],mu[:,21:]],1)
  return result,sigma,value,states
 agent.model.a2c_network.register_forward_hook(target_skip)
def predict(ob):return agent.model({'is_train':False,'prev_actions':None,'obs':ob,'rnn_states':None})

if args.collect:
 from expert_feedback import Expert
 if args.checkpoint:agent.restore(args.checkpoint,set_epoch=False);agent.model.eval()
 expert=Expert(hand,env,model,root0,rootq,limits,fids,wids,dev,dt,quat_apply,quat_apply_inverse)
 ob=env.reset();observations=[];labels=[];telemetry=[]
 for step in range(args.eval_steps):
  action=expert.action();observations.append(ob.cpu());labels.append(action.cpu())
  applied=action.clone()
  if args.checkpoint and args.teacher_mix<1:
   with torch.no_grad():learner=predict(ob)['mus'].clamp(-1,1)
   applied=args.teacher_mix*action+(1-args.teacher_mix)*learner
  # Brief target bias creates physical off-trajectory states; labels retain expert corrections.
  if 540<=step<600:
   phase=math.sin(math.pi*(step-540)/60)
   directions=(torch.arange(N,device=dev)%2*2-1)[:,None]
   applied[:,:21]+=directions*.012*phase/(limits[fids,1]-limits[fids,0])
   applied[:,21:23]+=directions*.0005/.25*phase
  ob,_,_,_=env.step(applied)
  if step%300==0:print('COLLECT',step,'holding',int((env.hold>=3).sum()),flush=True)
  if step%60==0:telemetry.append(dict(step=step,hold=env.hold.cpu().tolist(),force_N=env.contact.norm(dim=-1).cpu().tolist(),stable=env.stable.cpu().tolist(),lift=expert.lift.cpu().tolist(),depth=expert.depth.cpu().tolist(),object_state=obj.data.root_state_w.cpu().tolist()))
 success=(env.hold>=3)&(env.episode_falls==0)
 data=dict(obs=torch.stack(observations)[:,success.cpu()].reshape(-1,env.observation_space.shape[0]),actions=torch.stack(labels)[:,success.cpu()].reshape(-1,24),cases=env.cases,passed=success.cpu(),scope='Physical teacher trajectories with temporary target bias and external force/torque; only completed successful trajectories enter BC')
 torch.save(data,out/'demonstrations.pt')
 (out/'collection.json').write_text(json.dumps(dict(successes=int(success.sum()),num_envs=N,passed=success.cpu().tolist(),cases=env.cases.tolist(),diagnostics=env.diagnostics(),telemetry=telemetry),indent=2))
 print('COLLECTION_DONE',int(success.sum()),flush=True);os._exit(0)

def load_demonstrations():
 root=BASE.parent/'wrap_robustness_20261021'
 results=json.loads((root/'perturbation_results.json').read_text())
 heldout={'perturb_x_plus_05','perturb_y_minus_05','perturb_y_plus_05','perturb_yaw_plus_2','overlap_mixed_minus'}
 obs_rows=[];action_rows=[];sources=[]
 for result in results:
  if not result['passed'] or result['run'] in heldout:continue
  folder=root/result['run'];trace=json.loads((folder/'trace.json').read_text());audit=json.loads((folder/'audit.json').read_text())
  assert audit['joint_names']==hand.joint_names
  previous=np.array(audit['initial_q']);previous_wrist=np.zeros(3);count=0;recovery=0
  filtered=np.zeros(5);stable=0.
  lo=limits[:,0].cpu().numpy();hi=limits[:,1].cpu().numpy();fi=np.array(fids)
  for r in trace:
   if r['t']>=result['lift_time_s']+10:break
   q=np.array(r['q']);target=np.array(r['target']);wrist=np.array(r['wrist_target']);state=np.array(r['object'])
   state[:3]=(state[:3]-[0,0,.3])/.05;state[7:10]/=.1;state[10:13]/=2
   ob=np.r_[2*(q[fi]-lo[fi])/(hi[fi]-lo[fi])-1,2*(previous[fi]-lo[fi])/(hi[fi]-lo[fi])-1,np.array(r['wrist_actual'])/.05,previous_wrist/.05,state,np.array(r['normal_force_N'])/2,r['t']/25]
   if args.history:
    normal=np.array(r['normal_force_N']);filtered=.7*filtered+.3*normal
    ready=filtered[0]>.7 and filtered[1:].sum()>.3 and np.all(np.array(r['force_norm_N'])[1:]>.1) and normal.max()<4
    stable=stable+dt if ready else max(0,stable-dt)
    ob=np.r_[ob,filtered/2,min(1,stable/.75)]
   act=np.r_[2*(target[fi]-lo[fi])/(hi[fi]-lo[fi])-1,wrist/.25]
   obs_rows.append(ob);action_rows.append(act);previous=target;previous_wrist=wrist;count+=1
   recovery+=int(r['t']>=result['lift_time_s']+6)
  sources.append(dict(run=result['run'],samples=count,disturbance_and_recovery_samples=recovery,trace_sha256=hashlib.sha256((folder/'trace.json').read_bytes()).hexdigest()))
 dataset=dict(sources=sources,heldout_runs=sorted(heldout),total_samples=len(obs_rows),scope='Seven successful physical trajectories including force/torque disturbances and recovery; five successful poses excluded from BC')
 (out/'dataset.json').write_text(json.dumps(dataset,indent=2))
 ob=torch.tensor(np.array(obs_rows),device=dev,dtype=torch.float32).clamp(-5,5);ac=torch.tensor(np.array(action_rows),device=dev,dtype=torch.float32).clamp(-1,1)
 global bc_original_count
 bc_original_count=len(ob)
 if args.extra_data:
  extra=torch.load(args.extra_data,map_location=dev,weights_only=False)
  if args.only_extra:ob=extra['obs'].to(dev);ac=extra['actions'].to(dev)
  else:ob=torch.cat([ob,extra['obs'].to(dev)]);ac=torch.cat([ac,extra['actions'].to(dev)])
  dataset['extra_data']=args.extra_data;dataset['extra_samples']=len(extra['obs']);dataset['extra_sha256']=hashlib.sha256(Path(args.extra_data).read_bytes()).hexdigest();(out/'dataset.json').write_text(json.dumps(dataset,indent=2))
  dataset['only_extra']=bool(args.only_extra);dataset['effective_samples']=len(ob);(out/'dataset.json').write_text(json.dumps(dataset,indent=2))
 return ob,ac

def clear_diagnostics():
 env.max_speed=0.;env.max_force=0.;env.max_limit=0.;env.falls=0;env.episodes=0;env.anomalies=[]
def evaluate(label):
 clear_diagnostics()
 agent.model.eval();ob=env.reset();start_falls=env.falls;telemetry=[]
 if args.sample_eval:torch.manual_seed(args.action_seed)
 for step in range(args.eval_steps):
  with torch.no_grad():act=predict(ob)['actions' if args.sample_eval else 'mus']
  ob,_,_,_=env.step(act)
  if step%300==0:
   print('EVAL',label,step,'ever3s',int((env.best>=3).sum()),'holding',int((env.hold>=3).sum()),flush=True)
   telemetry.append(dict(step=step,stable=env.stable.cpu().tolist(),force_N=env.contact.norm(dim=-1).cpu().tolist(),wrist=env.wrist.cpu().tolist(),rise=(obj.data.root_pos_w[:,2]-.3).cpu().tolist()))
   (out/(label+'_progress.json')).write_text(json.dumps(telemetry,indent=2))
 success=(env.best>=3)&(env.hold>=3)&(env.episode_falls==0)
 row=dict(action_mode='sampled' if args.sample_eval else 'mean',action_seed=args.action_seed if args.sample_eval else None,telemetry=telemetry,target_skip=args.target_skip,finger_skip=args.finger_skip,label=label,num_envs=N,steps=args.eval_steps,successes=int(success.sum()),ever_held_3s=int((env.best>=3).sum()),falls=env.falls-start_falls,final_hold_s=env.hold.cpu().tolist(),best_hold_s=env.best.cpu().tolist(),passed=success.cpu().tolist(),cases=env.cases.tolist(),push=bool(args.push),diagnostics=env.diagnostics())
 (out/(label+'.json')).write_text(json.dumps(row,indent=2));return row

if args.checkpoint:agent.restore(args.checkpoint,set_epoch=False)
# Checkpoint optimizer state may carry the old 5e-4 learning rate.
agent.last_lr=pc['learning_rate']
for group in agent.optimizer.param_groups:group['lr']=pc['learning_rate']
if hasattr(agent.scheduler,'max_lr'):agent.scheduler.max_lr=pc['learning_rate']
if args.stable_kl:
 agent.scheduler.min_lr=1e-9;agent.scheduler.max_lr=1e-6
(out/'runtime_optimizer.json').write_text(json.dumps(dict(last_lr=agent.last_lr,optimizer_lr=[g['lr'] for g in agent.optimizer.param_groups],scheduler_max_lr=getattr(agent.scheduler,'max_lr',None)),indent=2))
if (not args.checkpoint or args.refit) and not args.retrieval_bc and not args.residual_bc:
 bc_obs,bc_actions=load_demonstrations();agent.model.eval()
 # Static scaling is deliberate: never re-estimate feature normalization during PPO.
 optimizer=torch.optim.Adam(agent.model.parameters(),lr=args.bc_lr)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.bc_steps,eta_min=1e-5)
 weight=torch.tensor([2500. if args.finger_skip else 1.]*21+[20.]*3,device=dev)
 for iteration in range(args.bc_steps):
  ids=torch.randint(len(bc_obs),(1024,),device=dev)
  if args.extra_fraction>=0:
   assert args.extra_data and not args.only_extra
   ids=torch.randint(bc_original_count,(1024,),device=dev);select=torch.rand(1024,device=dev)<args.extra_fraction
   ids[select]=torch.randint(bc_original_count,len(bc_obs),(int(select.sum()),),device=dev)
  ob=bc_obs[ids].clone()
  # Small sensor noise supplements physical recovery data; it is not counted as a new demonstration.
  if not args.finger_skip:ob[:,:42]+=torch.randn_like(ob[:,:42])*args.obs_noise
  ob[:,48:51]+=torch.randn_like(ob[:,48:51])*.001
  if args.recovery_augmentation:
   ob[:,21:42]+=torch.randn_like(ob[:,21:42])*.01
   ob[:,45:48]+=torch.randn_like(ob[:,45:48])*.04
  mu=predict(ob)['mus'];loss=((mu-bc_actions[ids]).square()*weight).mean()
  if args.contraction:
   perturbed=ob.clone();dq=torch.randn_like(ob[:,:21])*.001
   dq*=((ob[:,:21].abs()<.998)&(ob[:,21:42].abs()<.998))
   dw=torch.randn_like(ob[:,42:45])*.001
   perturbed[:,:21]+=dq;perturbed[:,21:42]+=dq;perturbed[:,42:45]+=dw;perturbed[:,45:48]+=dw
   expected=.8*torch.cat([dq,.2*dw],1)
   response=predict(perturbed)['mus']-mu
   loss+=args.contraction*((response-expected).square()*weight).mean()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.model.parameters(),1.);optimizer.step();scheduler.step()
  if iteration%500==0:print('BC',iteration,float(loss),flush=True)
 with torch.no_grad():err=predict(bc_obs)['mus']-bc_actions
 fit=dict(mse=float(err.square().mean()),joint_target_rmse_rad=float((err[:,:21]*(limits[fids,1]-limits[fids,0])/2).square().mean().sqrt()),wrist_target_rmse_mm=float((err[:,21:]*250).square().mean().sqrt()))
 (out/'bc_fit.json').write_text(json.dumps(fit,indent=2));print('BC_FIT',fit,flush=True)

# Exploration is per-channel, fixed for this short diagnostic and expressed in physical target units.
# The full original action ranges remain accessible through the policy mean.
with torch.no_grad():agent.model.a2c_network.sigma.copy_(torch.log(torch.tensor([args.finger_std]*21+[args.wrist_std]*3,device=dev)))
agent.model.a2c_network.sigma.requires_grad_(False)
if args.mode!='eval':agent.save(str(out/'bc_initial'))
if args.baseline_report:
 baseline_path=Path(args.baseline_report);previous_args=json.loads((baseline_path.parent/'execution.json').read_text())['args']
 for key in ('checkpoint','num_envs','seed','push','source','plans','pad_face','kp','kd','contact_k','contact_d','dt','mass','hand_gravity','eval_steps','history','target_skip','finger_skip','retrieval_bc','no_policy_time','no_target_input'):
  if key=='checkpoint' and args.residual_bc:
   assert args.bc_prior and not args.checkpoint
   old_state=torch.load(previous_args[key],map_location='cpu',weights_only=False)['model'];new_state=torch.load(args.bc_prior,map_location='cpu',weights_only=False)['model']
   keys=[k for k in old_state if 'actor_mlp.' in k or '.mu.' in k]
   assert keys and all(torch.equal(old_state[k],new_state[k]) for k in keys)
  else:
   assert previous_args.get(key,False if isinstance(vars(args).get(key),bool) else None)==vars(args).get(key),(key,previous_args.get(key),vars(args).get(key))
 before=json.loads(baseline_path.read_text());assert np.allclose(before['cases'],env.cases)
 before['reused_from']=str(baseline_path);(out/'before_ppo.json').write_text(json.dumps(before,indent=2))
else:before=evaluate('before_ppo')
if args.mode=='ppo':
 if args.diagnostic_ppo:
  if not before['passed'][0] or before['falls'] or before['successes']<int(.65*N):raise RuntimeError('Diagnostic baseline is not stable enough; stop')
 elif before['successes']<math.ceil(.9*N):raise RuntimeError('BC closed-loop gate failed; do not run PPO')
 if args.critic_warmup:
  from critic_warmup import install
  assert args.epochs>args.critic_warmup
  install(agent,args.critic_warmup,args.critic_lr,args.ppo_lr,out)
 if args.bc_anchor:
  from bc_anchor import install
  anchor_reference=install(agent,args.bc_anchor)
 env.reset();clear_diagnostics();agent.model.train();agent.train();agent.save(str(out/'ppo_final'));(out/'training_diagnostics.json').write_text(json.dumps(env.diagnostics(),indent=2));after=evaluate('after_ppo')
else:after=None
report=dict(status='completed',mode=args.mode,before=before,after=after,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),action_contract='absolute joint target normalized by original limits, absolute world wrist target / 0.25 m; original slew limits',observation_contract='fixed physical scaling, identical BC/PPO encoder',exploration_std=[args.finger_std]*21+[args.wrist_std]*3,config_changes={k:{'before':original_config['params']['config'].get(k),'after':v} for k,v in pc.items() if original_config['params']['config'].get(k)!=v},separate_actor_critic=True)
(out/'report.json').write_text(json.dumps(report,indent=2));print('FINISHED',args.mode,before['successes'],None if after is None else after['successes'],flush=True);os._exit(0)
