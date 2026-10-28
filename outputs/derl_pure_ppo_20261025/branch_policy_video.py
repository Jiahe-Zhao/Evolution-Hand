"""Opt-in frozen-policy recording, with native termination and unchanged physics."""
import os,json
import numpy as np
import torch
from native_goal_ppo import Policy,Moments,save_json

def setup(e):
 import isaaclab.sim as sim
 from isaaclab.sensors import Camera,CameraCfg
 e.review_camera=Camera(CameraCfg(prim_path='/World/ReviewCamera',height=608,width=800,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=26.,clipping_range=(.001,10.))))
 # Visual-only ground: no collision or rigid-body configuration.
 floor=sim.CuboidCfg(size=(1.,1.,.005),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.12,.16,.22),roughness=.9))
 floor.func('/World/ReviewGround',floor,translation=(0,0,.21))

def record(env,args,out):
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw
 e=env.env;dev=e.device;camera=e.review_camera
 import isaaclab.sim as sim
 from pxr import Usd,UsdShade
 appearance={}
 for role,color in {'pad':(.06,.46,.28),'bearing':(.6,.32,.08),'body':(.22,.31,.42)}.items():
  mat=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.65);mat.func('/World/ReviewAppearance/'+role,mat)
  appearance[role]=UsdShade.Material(e.sim.stage.GetPrimAtPath('/World/ReviewAppearance/'+role))
 for prim in Usd.PrimRange(e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
  if prim.GetTypeName()=='Mesh' and '/visuals/' in str(prim.GetPath()):
   name=str(prim.GetPath());role='pad' if 'finger_pad' in name else ('bearing' if 'bearing' in name or 'shaft' in name else 'body')
   UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance[role],bindingStrength='strongerThanDescendants',materialPurpose='allPurpose')
 path=os.environ['BRANCH_POLICY_VIDEO'];ck=torch.load(path,map_location=dev,weights_only=False)
 net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(dev);net.load_state_dict(ck['policy']);net.eval()
 rms=Moments((env.observation_space.shape[0],),dev)
 for k,v in ck['observation_moments'].items():setattr(rms,k,v)
 env.evaluating=True;results=[]
 for seed in [1062802,1062803,1062804]:
  env.reset_seed=seed;obs=env.reset()
  target=torch.tensor([[0.,-.005,.375]],device=dev)
  camera.set_world_poses_from_view(target+torch.tensor([[.29,-.34,.22]],device=dev),target)
  video=out/f'branch_policy_seed{seed}.mp4';writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=8)
  success=False;last=None;forces=[]
  for t in range(e.max_episode_length):
   if t%3==0:
    e.sim.render();camera.update(1/20,force_recompute=True)
    im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));d=ImageDraw.Draw(im)
    d.rectangle((0,0,800,52),fill=(18,25,38));d.text((12,8),f'Branch | trained PPO | mean action | {t/60:.2f} s',fill='white')
    d.text((12,29),'Frozen observation statistics | original native success / termination',fill='white')
    last=np.asarray(im).copy();writer.append_data(last)
   with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.mean
   obs,reward,done,info=env.step(action)
   forces.append(e.whole_hand_tactile.digit_forces()[0].tolist())
   if bool(done[0]):
    success=bool(info['episode_success'][0]);break
  if last is not None:
   im=Image.fromarray(last);d=ImageDraw.Draw(im);d.rectangle((0,552,800,608),fill=(18,25,38));d.text((12,565),f'Native success: {success} | episode ended at {(t+1)/60:.2f}s',fill='white');d.text((12,585),'End card: frozen last pre-reset frame (not extended holding)',fill='white')
   for _ in range(20):writer.append_data(np.asarray(im))
  writer.close();results.append(dict(seed=seed,video=str(video),success=success,steps=t+1,duration_s=(t+1)/60,digit_forces_N=forces))
  save_json(out/'video_results.json',dict(checkpoint=path,checkpoint_frames=ck.get('frames'),action_mode='mean',frozen_normalization=True,native_termination=True,results=results))
  if success:break
 save_json(out/'completed.json',dict(status='completed',successful_videos=sum(r['success'] for r in results)))
