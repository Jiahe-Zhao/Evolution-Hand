"""Live Isaac camera recording; colors affect appearance only."""
import numpy as np
import torch
from pxr import Usd, UsdShade
import isaaclab.sim as sim
from isaaclab.sensors import Camera, CameraCfg

class Recorder:
 def __init__(self,ctx,out,device):
  import imageio.v2 as imageio
  self.writer=imageio.get_writer(str(out/'test.mp4'),fps=30,codec='libx264',quality=8)
  self.ctx=ctx
  light=sim.DomeLightCfg(intensity=900.,color=(.65,.72,.85),visible_in_primary_ray=False);light.func('/World/VideoLight',light)
  colors={'pad':(.08,.65,.40),'metal':(.72,.38,.08)}
  materials={}
  for role,color in colors.items():
   cfg=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.45)
   cfg.func('/World/VideoMaterials/'+role,cfg);materials[role]=UsdShade.Material(ctx.stage.GetPrimAtPath('/World/VideoMaterials/'+role))
  for prim in Usd.PrimRange(ctx.stage.GetPrimAtPath('/World/Hand')):
   if prim.GetTypeName()=='Mesh':
    role='pad' if 'finger_pad' in str(prim.GetPath()) else 'metal'
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role],bindingStrength='strongerThanDescendants')
  self.camera=Camera(CameraCfg(prim_path='/World/VideoCamera',height=720,width=960,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=24.,horizontal_aperture=20.955,clipping_range=(.001,10.))))
  self.device=device
 def initialize(self):
  self.camera.set_world_poses_from_view(torch.tensor([[.10,-.035,.185]],device=self.device),torch.tensor([[0.,0.,.145]],device=self.device))
  for _ in range(5):self.ctx.render()
 def frame(self,t,mode,forces,released,slip):
  from PIL import Image,ImageDraw
  self.ctx.render();self.camera.update(1/30,force_recompute=True)
  data=self.camera.data.output['rgb'][0].cpu().numpy()[:,:,:3]
  im=Image.fromarray(data.astype(np.uint8));draw=ImageDraw.Draw(im)
  draw.rectangle((0,0,960,87),fill=(18,25,38))
  draw.text((16,10),f'BASELINE CONTACT BENCH | {mode.upper()} | LIVE ISAAC SIM | t={t:.2f} s',fill='white')
  draw.text((16,32),f'Left {forces["left"]:.3f} N   Right {forces["right"]:.3f} N   '+('SUPPORT REMOVED' if released else 'APPROACH / CONTACT'),fill=(100,230,180))
  draw.text((16,54),f'Green: fingertip pad | Orange: bearings | Blue: object | slip {slip*1000:.4f} mm',fill='white')
  self.writer.append_data(np.asarray(im))
 def close(self):self.writer.close()
