"""Opt-in frozen-policy recording, with native termination and unchanged physics."""
import os,json
import numpy as np
import torch
from native_goal_ppo import Policy,Moments,save_json

def setup(e):
 import isaaclab.sim as sim
 from isaaclab.sensors import Camera,CameraCfg
 e.review_camera=Camera(CameraCfg(prim_path='/World/ReviewCamera',height=608,width=800,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=26.,clipping_range=(.001,10.))))

def install(env_id):
 import gymnasium as gym,importlib
 module,name=gym.spec(env_id).entry_point.split(':');cls=getattr(importlib.import_module(module),name);old=cls._setup_scene
 def scene(e):
  old(e);setup(e)
 cls._setup_scene=scene

def record(env,args,out):
 import imageio.v2 as imageio
 from PIL import Image,ImageDraw
 e=env.env.unwrapped;dev=e.device;camera=e.review_camera
 import isaaclab.sim as sim
 from pxr import Usd,UsdShade
 appearance={}
 for role,color in {'pad':(.06,.46,.28),'bearing':(.6,.32,.08),'body':(.22,.31,.42)}.items():
  mat=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.65);mat.func('/World/ReviewAppearance/'+role,mat)
  appearance[role]=UsdShade.Material(e.sim.stage.GetPrimAtPath('/World/ReviewAppearance/'+role))
 root=e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')
 # Visual instances may have been reintroduced by scene cloning.
 for _ in range(4):
  instances=[p for p in Usd.PrimRange(root) if p.IsInstance() and '/visuals' in str(p.GetPath())]
  for p in instances:p.SetInstanceable(False)
 from pxr import UsdPhysics
 audit=[]
 for prim in Usd.PrimRange(root):
  binding=prim.GetRelationship('material:binding')
  if binding and binding.GetTargets():
   target=e.sim.stage.GetPrimAtPath(binding.GetTargets()[0])
   if target.HasAPI(UsdPhysics.MaterialAPI):
    api=UsdShade.MaterialBindingAPI.Apply(prim)
    if not prim.GetRelationship('material:binding:physics').GetTargets():
     api.Bind(UsdShade.Material(target),materialPurpose='physics')
    prim.RemoveProperty('material:binding')
  if prim.GetTypeName()=='Mesh' and '/visuals/' in str(prim.GetPath()):
   name=str(prim.GetPath());role='pad' if 'finger_pad' in name else ('bearing' if 'bearing' in name or 'shaft' in name else 'body')
   UsdShade.MaterialBindingAPI.Apply(prim).Bind(appearance[role],bindingStrength='strongerThanDescendants')
   audit.append(dict(path=name,role=role))
 save_json(out/'visual_audit.json',audit)
 if not audit:raise RuntimeError('No mechanical visual meshes recolored')
 path=os.environ['FORAGE_POLICY_VIDEO'];ck=torch.load(path,map_location=dev,weights_only=False)
 net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(dev);net.load_state_dict(ck['policy']);net.eval()
 rms=Moments((env.observation_space.shape[0],),dev)
 for k,v in ck['observation_moments'].items():setattr(rms,k,v)
 env.evaluating=True;results=[];mode='sample';action_seed=int(os.environ.get('FORAGE_VIDEO_ACTION_SEED','1062803'))
 torch.manual_seed(action_seed)
 for seed in json.loads(os.environ.get('FORAGE_VIDEO_SCENES','[1062802,1062809]')):
  env.reset_seed=seed;obs=env.reset();torch.manual_seed(action_seed)
  target=torch.tensor([[0.,0.,.14]],device=dev)
  camera.set_world_poses_from_view(target+torch.tensor([json.loads(os.environ.get('FORAGE_VIDEO_EYE','[.28,-.35,.32]'))],device=dev),target)
  video=out/f'forage_action{action_seed}_scene{seed}.mp4';writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=8)
  success=False;last=None;forces=[]
  for t in range(e.max_episode_length):
   if t%3==0:
    e.sim.render();camera.update(1/20,force_recompute=True)
    im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));d=ImageDraw.Draw(im)
    d.rectangle((0,0,800,52),fill=(18,25,38));d.text((12,8),f'Forage | trained PPO | {mode} action | {t/60:.2f} s',fill='white')
    d.text((12,29),'Frozen observation statistics | original native success / termination',fill='white')
    last=np.asarray(im).copy();writer.append_data(last)
   with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.sample() if mode=='sample' else dist.mean
   obs,reward,done,info=env.step(action)
   forces.append(e.whole_hand_tactile.digit_forces()[0].tolist())
   if bool(done[0]):
    success=bool(info['episode_success'][0]);break
  if last is not None:
   im=Image.fromarray(last);d=ImageDraw.Draw(im);d.rectangle((0,552,800,608),fill=(18,25,38));d.text((12,565),f'Native success: {success} | episode ended at {(t+1)/60:.2f}s',fill='white');d.text((12,585),'End card: frozen last pre-reset frame (not extended holding)',fill='white')
   for _ in range(20):writer.append_data(np.asarray(im))
  writer.close();results.append(dict(seed=seed,video=str(video),success=success,steps=t+1,duration_s=(t+1)/60,digit_forces_N=forces))
  save_json(out/'video_results.json',dict(checkpoint=path,checkpoint_frames=ck.get('frames'),action_mode=mode,action_seed=action_seed,frozen_normalization=True,native_termination=True,results=results))

 save_json(out/'completed.json',dict(status='completed',successful_videos=sum(r['success'] for r in results)))
