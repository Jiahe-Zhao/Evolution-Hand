"""Frozen policy video evaluation; all attempted episode outcomes retained."""
import json, hashlib
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from goal_ppo import Policy,Moments,clear_diagnostics,save_json

def record(env,args,out,g):
 checkpoint=torch.load(args.checkpoint,map_location=g['dev'],weights_only=False)
 net=Policy(152).to(g['dev']);net.load_state_dict(checkpoint['policy']);net.eval()
 rms=Moments((152,),g['dev'])
 for k,v in checkpoint['observation_moments'].items():setattr(rms,k,v)
 env.evaluating=True
 camera=g['review_camera'];results=[]
 for index,height in enumerate((.04,.055,.025)):
  clear_diagnostics(env);obs=env.reset();env.goals.target.fill_(height);obs=env.obs()
  video=out/f'grasp_goal_{round(height*1000)}mm.mp4'
  writer=imageio.get_writer(str(video),fps=20,codec='libx264',quality=8)
  target=torch.tensor([[0.,0.,.34]],device=g['dev'])
  camera.set_world_poses_from_view(target+torch.tensor([[.35,-.40,.23]],device=g['dev']),target)
  for t in range(args.episode_steps):
   with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.mean
   obs,_,_,_=env.step(action)
   if t%3==0:
    g['ctx'].render();camera.update(1/20,force_recompute=True)
    im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));d=ImageDraw.Draw(im)
    rise=float(g['obj'].data.root_pos_w[0,2]-g['initial_object'][0,2])
    d.rectangle((0,0,800,62),fill=(18,25,38));d.text((12,8),f'Pure PPO | mean action | target {height*1000:.0f} mm | {(t+1)/60:.2f} / 25 s',fill='white')
    d.text((12,28),f'lift {rise*1000:.1f} mm | hold {float(env.hold[0]):.2f} s | falls {int(env.episode_falls[0])}',fill='white')
    d.text((12,46),'External push at 18.0-18.5 s | frozen normalization | no BC',fill='white');writer.append_data(np.asarray(im))
  writer.close()
  passed=bool((env.hold[0]>=3)&(env.episode_falls[0]==0))
  results.append(dict(video=str(video),success=passed,target_m=height,final_hold_s=float(env.hold[0]),falls=int(env.episode_falls[0]),diagnostics=env.diagnostics()))
  save_json(out/'video_results.json',dict(checkpoint=args.checkpoint,frames=checkpoint['frames'],action_mode='mean',results=results))
  print('VIDEO',video,passed,flush=True)
 save_json(out/'completed.json',dict(status='completed',successful_videos=sum(r['success'] for r in results)))
