"""Full hand physical motion test with the production import and actuator limits."""
import os,sys,argparse,json,math
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/zjh/Evolution_PC/Isaaclab_other')
sys.path.insert(0,'/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old=fi._fast_walk
def walk(path,*a,**k):
    if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
    yield from old(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser();p.add_argument('--urdf',required=True);p.add_argument('--output',required=True);p.add_argument('--record',action='store_true');p.add_argument('--full-range',action='store_true');p.add_argument('--dt',type=float,default=1/240)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=args.enable_cameras or args.record
app=AppLauncher(args).app
import torch,numpy as np,importlib.util
import isaaclab.sim as sim
from isaaclab.assets import Articulation
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
sys.path.insert(0,str(BASE/'patch'))
from mechanical_contact_materials import configure
from pxr import Usd,UsdShade
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
ctx=sim.SimulationContext(sim.SimulationCfg(dt=args.dt,device=args.device,gravity=(0,0,-9.81)))
cfgpath=out/'hand_cfg.py';parse_urdf_and_generate_articulation_cfg(args.urdf,args.urdf,str(cfgpath))
spec=importlib.util.spec_from_file_location('motion_cfg',cfgpath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cfg=module.CURRENT_HAND_CFG.replace(prim_path='/World/Hand');cfg.init_state.pos=(0,0,.3)
cfg.spawn.make_instanceable=False;cfg.spawn.usd_dir=str(out/'usd');cfg.spawn.force_usd_conversion=True
hand=Articulation(cfg);materials=configure(ctx.stage,'/World/Hand')
camera=writer=None
if args.record:
    import imageio.v2 as imageio
    from isaaclab.sensors import Camera,CameraCfg
    light=sim.DomeLightCfg(intensity=850.,color=(.72,.78,.9),visible_in_primary_ray=False);light.func('/World/Light',light)
    floor=sim.GroundPlaneCfg(color=(.16,.19,.23));floor.func('/World/Ground',floor)
    appearance={}
    for role,color in {'pad':(.08,.55,.32),'bearing':(.65,.36,.10),'body':(.28,.35,.42)}.items():
        material=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.55);material.func('/World/Appearance/'+role,material)
        appearance[role]=UsdShade.Material(ctx.stage.GetPrimAtPath('/World/Appearance/'+role))
    for prim in Usd.PrimRange(ctx.stage.GetPrimAtPath('/World/Hand')):
        if prim.GetTypeName()=='Mesh':
            name=str(prim.GetPath());role='pad' if 'finger_pad' in name else ('bearing' if 'bearing' in name or 'shaft' in name else 'body')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance[role],bindingStrength='strongerThanDescendants')
    camera=Camera(CameraCfg(prim_path='/World/Camera',height=600,width=800,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=30.,clipping_range=(.001,10.))))
    writer=imageio.get_writer(str(out/'motion.mp4'),fps=24,codec='libx264',quality=7)
ctx.reset();hand.update(args.dt)
if camera:
    target=hand.data.body_pos_w[0].mean(dim=0,keepdim=True)
    camera.set_world_poses_from_view(target+torch.tensor([[.20,-.24,.22]],device=args.device),target)
    for _ in range(5):ctx.render()
limits=hand.root_physx_view.get_dof_limits().to(hand.device)[0];zero=torch.clamp(torch.zeros(len(hand.joint_names),device=hand.device),limits[:,0],limits[:,1])
hand.write_joint_state_to_sim(zero[None],torch.zeros_like(zero)[None]);hand.set_joint_position_target(zero[None])
targets=[zero.clone()]
spread=zero.clone();curl=zero.clone()
for i,name in enumerate(hand.joint_names):
    if 'mcp_spread' in name:spread[i]=min(.18,float(limits[i,1])*.7)
    elif 'thumb_spread' in name:spread[i]=min(.45,float(limits[i,1])*.5)
    else:curl[i]=min(1.2 if args.full_range else .45,float(limits[i,1])*(.8 if args.full_range else .35))

for i,name in enumerate(hand.joint_names):
    if name.endswith("wrist_roll"): spread[i]=0.6
    elif name.endswith("wrist_pitch"): spread[i]=-0.45
    elif name.endswith("wrist_yaw"): spread[i]=0.7
targets.extend([spread,zero.clone(),curl,curl+spread*.5,zero.clone()])
phase_duration=6. if args.full_range else 3.;total=(len(targets)-1)*phase_duration+1.;trace=[];peak_limit=peak_velocity=peak_target_velocity=peak_error=0.;last_target=zero.clone()
for step in range(round(total/args.dt)):
    t=step*args.dt
    phase=min(int(t/phase_duration),len(targets)-2);u=min(1.,(t-phase*phase_duration)/phase_duration)
    s=u*u*u*(10-15*u+6*u*u)
    q=targets[phase]*(1-s)+targets[phase+1]*s
    hand.set_joint_position_target(q[None]);hand.write_data_to_sim();ctx.step(render=False);hand.update(args.dt)
    positions=hand.data.joint_pos[0];velocities=hand.data.joint_vel[0]
    excess=torch.maximum(limits[:,0]-positions,positions-limits[:,1]).clamp_min(0)
    peak_limit=max(peak_limit,float(excess.max()));peak_velocity=max(peak_velocity,float(abs(velocities).max()));peak_error=max(peak_error,float(abs(q-positions).max()));peak_target_velocity=max(peak_target_velocity,float(abs(q-last_target).max())/args.dt);last_target=q.clone()
    if not bool(torch.isfinite(positions).all()):raise RuntimeError('nonfinite dynamics')
    if step%max(1,round(1/(24*args.dt)))==0:
        trace.append({'t':t,'target':q.cpu().tolist(),'q':positions.cpu().tolist(),'qd':velocities.cpu().tolist(),'torque':hand.data.applied_torque[0].cpu().tolist()})
        if camera:
            from PIL import Image,ImageDraw
            ctx.render();camera.update(1/24,force_recompute=True);im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));draw=ImageDraw.Draw(im);draw.rectangle((0,0,800,48),fill=(20,27,38));draw.text((12,10),f'LIVE ISAAC | morphology contact v3 | t={t:.2f}s | phase={phase+1}',fill='white');draw.text((12,28),f'Self-collision ON | joint limit excess {float(excess.max()):.5f} rad',fill='white');writer.append_data(np.asarray(im))
if writer:writer.close()
report={'source':args.urdf,'dt':args.dt,'joint_names':hand.joint_names,'peak_limit_excess_rad':peak_limit,'peak_joint_velocity_rad_s':peak_velocity,'peak_target_velocity_rad_s':peak_target_velocity,'peak_tracking_error_rad':peak_error,'self_collision_enabled':True,'target_interpolation':'quintic, continuous position/velocity/acceleration','materials':materials,'masses':hand.root_physx_view.get_masses().cpu().tolist(),'passed':peak_limit<.03 and peak_velocity<1.,'scope':'motion and contact stability only; not grasp/Strike success'}
(out/'trace.json').write_text(json.dumps(trace));(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['materials','masses','joint_names']}),flush=True);os._exit(0)
