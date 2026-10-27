"""From-scratch DERL-style PPO on the existing physical grasp task."""
import os,sys,json,argparse,math,re,hashlib,shutil,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'wrap_learning_repair_20261022'))
sys.path.insert(0,str(BASE.parent/'contact_control_calibration_20261021'))
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old_walk=fi._fast_walk
def walk(path,*a,**k):
 if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
 yield from old_walk(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser()
p.add_argument('--source',required=True);p.add_argument('--plans',required=True);p.add_argument('--output',required=True)
p.add_argument('--seed',type=int,default=2027);p.add_argument('--eval-seed',type=int,default=92027)
p.add_argument('--num-envs',type=int,default=256);p.add_argument('--steps',type=int,default=5_000_000)
p.add_argument('--eval-every',type=int,default=1_000_000);p.add_argument('--horizon',type=int,default=128)
p.add_argument('--smoke',action='store_true')
p.add_argument('--goal-heights',type=float,nargs='+',default=[.025,.040,.055])
p.add_argument('--goal-tolerance',type=float,default=.0075)
p.set_defaults(plan_index=0,pad_face='ymixed',dt=1/240,kp=3.,kd=.3,contact_k=1500.,contact_d=10.,torsion_radius=0.,mass=.08,offset_mm=0.,offset_y_mm=0.,yaw_deg=0.,hand_gravity=True,episode_steps=1500,push=True,completion_reward=False)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.steps<args.num_envs*args.horizon:raise ValueError('Budget must cover one full rollout')
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copyfile(__file__,out/'worker_snapshot.py')
for name in ('goal_ppo.py','goal_contract.py','features.py'):shutil.copyfile(BASE/name,out/(Path(name).stem+'_snapshot.py'))
(out/'execution.json').write_text(json.dumps(dict(args=vars(args),argv=sys.argv,bc_updates=0,checkpoint_loaded=False,initialization='random actor and critic',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
from diagnostic_wrist import build,NAMES
source=Path(args.source).resolve();path,contract=build(source,out/'wrist')
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
row=feasible[args.plan_index]
# Recompute initialization validity on this morphology. A historical plan's
# stored collision list is not evidence for a mutated hand.
qplan=np.array([row['q'][name] for name in model.names])
if np.any(qplan<np.array(model.lo)) or np.any(qplan>np.array(model.hi)):
 raise ValueError('Initialization exceeds this morphology joint limits')
hits=model.collisions(qplan,np.array(row['center']),np.array(row['rotation']))
if any(h['depth_m']>1e-5 for h in hits):
 raise ValueError('Initialization intersects this morphology; replan, do not reject morphology fitness')
(out/'initialization_audit.json').write_text(json.dumps(dict(source=str(source),intersections=hits),indent=2))
G=np.array(row['rotation']).T;center=np.array([0.,0.,.3]);root_position=center+G@(model.root[:3,3]-np.array(row['center']));root_R=G@model.root[:3,:3];quat=Rotation.from_matrix(root_R).as_quat()[[3,0,1,2]]
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

pad_ids=[hand.body_names.index(name) for _,name,_,_ in model.pad_specs]
assert len(pad_ids)==5
pad_offsets=torch.tensor(np.array([point for _,_,point,_ in model.pad_specs]),device=dev,dtype=torch.float32)
object_half_size=torch.tensor([.009,.009,.09],device=dev)
from features import relative_features,object_contact_forces
from goal_contract import LiftGoals
class WrapEnv:
 def __init__(self):
  self.num_envs=N
  self.action_space=gym.spaces.Box(-1,1,(24,),dtype=np.float32)
  self.observation_space=gym.spaces.Box(-5,5,(152,),dtype=np.float32)
  self.goals=LiftGoals(N,dev,args.seed+100,args.goal_heights,args.goal_tolerance)
  self.target=initial_q.clone();self.wrist=torch.zeros((N,3),device=dev)
  self.age=torch.zeros(N,device=dev,dtype=torch.long);self.contact=torch.zeros((N,5,3),device=dev)
  self.support_force=torch.zeros(N,device=dev);self.hold=torch.zeros(N,device=dev);self.best=torch.zeros(N,device=dev)
  self.filtered=torch.zeros((N,5),device=dev);self.stable=torch.zeros(N,device=dev)
  self.episode_falls=torch.zeros(N,device=dev,dtype=torch.long)
  self.max_speed=0.;self.max_force=0.;self.max_limit=0.;self.falls=0;self.episodes=0;self.force_exceedances=0
  self.peak_lift=torch.zeros(N,device=dev);self.anomalies=[];self.reward_terms={}
  self.potential=torch.zeros(N,device=dev);self.evaluating=False
  self.shaping_return=torch.zeros(N,device=dev);self.discount_power=torch.ones(N,device=dev);self.successful_terminal_episodes=0;self.max_terminal_shaping_residual=0.
  fixed=[(0,0,0),(-.5,0,0),(.5,0,0),(0,-.5,0),(0,.5,0),(0,0,-2),(0,0,2),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-.5,-.5,-1),(.5,.5,1)]
  rng=np.random.default_rng(args.seed)
  self.cases=np.array([fixed[i] if i<len(fixed) else (*rng.uniform(-.5,.5,2),rng.uniform(-1.5,1.5)) for i in range(N)])
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
  parts.extend([(self.hold/3).clamp(0,1)[:,None],self.contact.reshape(N,15)/2,hand.data.joint_vel[:,fids]/.45,hand.data.body_lin_vel_w[:,root_id]/.1,self.support_force[:,None]/2])
  parts.append(relative_features(hand.data.body_pos_w[:,root_id],hand.data.body_quat_w[:,root_id],hand.data.body_lin_vel_w[:,root_id],hand.data.body_ang_vel_w[:,root_id],obj.data.root_state_w,self.contact))
  parts.append(self.pad_relative().flatten(1)/.05)
  parts.append(self.goals.features(obj.data.root_pos_w[:,2]-initial_object[:,2]))
  return torch.cat(parts,1).clamp(-5,5)
 def pad_relative(self):
  points=hand.data.body_pos_w[:,pad_ids]+quat_apply(hand.data.body_quat_w[:,pad_ids],pad_offsets[None].expand(N,-1,-1))
  return quat_apply_inverse(obj.data.root_quat_w[:,None,:].expand(-1,5,-1),points-obj.data.root_pos_w[:,None,:])
 def reset_ids(self,ids):
  if not len(ids):return
  if not self.evaluating:self.goals.reset(ids)
  state=initial_object[ids].clone()
  perturb=torch.tensor(self.cases[ids.cpu().numpy()],device=dev,dtype=torch.float32)
  state[:,:2]+=perturb[:,:2]/1000
  yaw=perturb[:,2]*math.pi/180
  state[:,3]=torch.cos(yaw/2);state[:,4:6]=0;state[:,6]=torch.sin(yaw/2)
  hand.write_joint_state_to_sim(initial_q[ids],torch.zeros_like(initial_q[ids]),env_ids=ids)
  obj.write_root_state_to_sim(state,env_ids=ids);hand.reset(ids);obj.reset(ids)
  self.target[ids]=initial_q[ids];self.wrist[ids]=0;self.age[ids]=0;self.contact[ids]=0;self.hold[ids]=0
  self.support_force[ids]=0;self.filtered[ids]=0;self.stable[ids]=0;self.potential[ids]=0;self.shaping_return[ids]=0;self.discount_power[ids]=1
  hand.set_joint_position_target(self.target);hand.write_data_to_sim();obj.write_data_to_sim();ctx.forward();hand.update(args.dt);obj.update(args.dt)
 def reset(self):
  if self.evaluating:self.goals.reset(torch.arange(N,device=dev),evaluation=True)
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
  self.support_force=support_force
  contact_local=object_contact_forces(obj.data.root_quat_w,self.contact)
  normal=contact_local[:,:,0].abs();norm=self.contact.norm(dim=-1)
  self.filtered=.7*self.filtered+.3*normal
  ready=(self.filtered[:,0]>.7)&(self.filtered[:,1:].sum(1)>.3)&(norm[:,1:]>.1).all(1)&(normal.max(1).values<4)
  self.stable=torch.where(ready,self.stable+dt,(self.stable-dt).clamp_min(0))
  touch=(normal[:,0]>.1)&(normal[:,1:].sum(1)>.1)
  upright=quat_apply(obj.data.root_quat_w,torch.tensor([0.,0.,1.],device=dev).expand(N,-1))[:,2]
  speed=obj.data.root_lin_vel_w.norm(dim=-1)
  held=(rise>.015)&(support_force<.05)&touch&(upright>.98)&(speed<.1)&self.goals.reached(rise)
  self.hold=torch.where(held,self.hold+dt,torch.zeros_like(self.hold));self.best=torch.maximum(self.best,self.hold);self.peak_lift=torch.maximum(self.peak_lift,rise)
  # Height only pays while grasped. Excess force and velocity are soft costs,
  # never episode terminations or hidden constraints on allowable morphology.
  balance=torch.minimum(normal[:,0]/1.5,normal[:,1:].sum(1)/1.5).clamp(0,1)
  opposed=(contact_local[:,0,0]*contact_local[:,1:,0].sum(1)<0).float()
  quality=balance*opposed*torch.exp(-torch.relu(speed-.08)*10)*upright.clamp(0,1)
  overforce=torch.relu(norm-4).square().sum(1).clamp_max(100)
  overspeed=torch.relu(speed-.15).square().clamp_max(25)
  distance=torch.relu(self.pad_relative().abs()-object_half_size).norm(dim=-1)
  reach=torch.exp(-distance/.02).mean(1)
  reward=reach+balance+5*self.goals.lift_reward(rise)*quality+3*held.float()-.1*overforce-2*overspeed
  self.reward_terms=dict(reach=float(reach.mean()),balance=float(balance.mean()),height=float((self.goals.lift_reward(rise)*quality).mean()),overforce=float(overforce.mean()),overspeed=float(overspeed.mean()))
  if not torch.isfinite(hand.data.joint_pos).all() or not torch.isfinite(obj.data.root_state_w).all():raise RuntimeError('Nonfinite physical state')
  self.max_limit=max(self.max_limit,float(torch.maximum(limits[:,0]-hand.data.joint_pos,hand.data.joint_pos-limits[:,1]).clamp_min(0).max()))
  self.max_speed=max(self.max_speed,float(speed.max()));self.max_force=max(self.max_force,float(norm.max()))
  self.force_exceedances+=int((norm.max(1).values>25).sum())
  bad=(norm.max(1).values>25)|((speed>2)&(obj.data.root_pos_w[:,2]>.2))
  if bad.any() and len(self.anomalies)<20:
   i=int(bad.nonzero()[0]);self.anomalies.append(dict(env=i,age=int(self.age[i]),force_N=norm[i].cpu().tolist(),object=obj.data.root_state_w[i].cpu().tolist(),q=hand.data.joint_pos[i].cpu().tolist()))
  fell=obj.data.root_pos_w[:,2]<.15;timeout=self.age>=args.episode_steps;done=fell|timeout
  completed=timeout&(self.hold>=3)&(~fell)
  self.successful_terminal_episodes+=int(completed.sum())
  reward=reward+10*completed.float()-10*fell.float()
  self.falls+=int(fell.sum());self.episodes+=int(done.sum());self.episode_falls+=fell.long()
  self.reset_ids((fell if self.evaluating else done).nonzero().flatten())
  return self.obs(),reward,done,{'episode_success':completed,'held':held,'rise':rise}
 def diagnostics(self):return dict(force_exceedances=self.force_exceedances,successful_terminal_episodes=self.successful_terminal_episodes,max_terminal_shaping_residual=self.max_terminal_shaping_residual,episode_steps=args.episode_steps,completion_reward=args.completion_reward,episodes=self.episodes,falls=self.falls,max_speed_m_s=self.max_speed,max_digit_force_N=self.max_force,max_limit_rad=self.max_limit,anomalies=self.anomalies)


from goal_ppo import run
try:
 run(WrapEnv(),args,out)
except Exception:
 import traceback
 error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True);os._exit(1)
os._exit(0)
