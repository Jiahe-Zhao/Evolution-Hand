"""Paired independent tests. No training or checkpoint mutation."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from derl_ppo import Policy,Moments,clear_diagnostics,save_json

def run(env,args,out):
 g=env.step.__globals__;device=g['dev'];N=env.num_envs
 checkpoints={name:torch.load(Path(args.checkpoint_dir)/(name+'.pth'),map_location=device,weights_only=False) for name in ('best','final')}
 net=Policy(148).to(device);rms=Moments((148,),device)
 camera=g.get('review_camera')
 if camera:
  target=torch.tensor([[0.,0.,.35]],device=device);camera.set_world_poses_from_view(target+torch.tensor([[.35,-.40,.25]],device=device),target)
  for _ in range(5):g['ctx'].render()
 tests=[('best_independent','best','best',1.,True),('final_independent','final','final',1.,True),('best_expanded','best','best',2.,True),('final_expanded','final','final',2.,True),('final_actor_best_norm','final','best',1.,True),('best_actor_final_norm','best','final',1.,True),('final_no_push','final','final',1.,False)]
 if args.video_only:tests=tests[:2]
 summary=[]
 for label,actor,norm,scale,push in tests:
  net.load_state_dict(checkpoints[actor]['policy']);net.eval()
  for k,v in checkpoints[norm]['observation_moments'].items():setattr(rms,k,v)
  rng=np.random.default_rng(1062601);env.cases=np.column_stack((rng.uniform(-.5,.5,(N,2)),rng.uniform(-1.5,1.5,N)))*scale
  args.push=push;clear_diagnostics(env);env.evaluating=True;obs=env.reset();traces=[];writer=None
  if camera and label.endswith('_independent'):
   import imageio.v2 as imageio
   from PIL import Image,ImageDraw
   writer=imageio.get_writer(str(out/(label+'.mp4')),fps=20,codec='libx264',quality=7)
  start=time.monotonic()
  for t in range(1500):
   with torch.no_grad():dist,_=net(rms.normalize(obs));action=dist.mean
   obs,reward,done,info=env.step(action)
   if t%3==0 or t==1499:
    upright=g['quat_apply'](g['obj'].data.root_quat_w,torch.tensor([0.,0.,1.],device=device).expand(N,-1))[:,2]
    traces.append(dict(t=(t+1)/60,action=action.cpu().numpy(),q=g['hand'].data.joint_pos.cpu().numpy(),wrist=env.wrist.cpu().numpy(),object=(g['obj'].data.root_pos_w-g['origins']).cpu().numpy(),speed=g['obj'].data.root_lin_vel_w.norm(dim=-1).cpu().numpy(),upright=upright.cpu().numpy(),force=env.contact.cpu().numpy(),support=env.support_force.cpu().numpy(),hold=env.hold.cpu().numpy(),falls=env.episode_falls.cpu().numpy(),reward=reward.cpu().numpy(),held=info['held'].cpu().numpy()))
    if writer:
     target=g['obj'].data.root_pos_w[0:1].clone();camera.set_world_poses_from_view(target+torch.tensor([[.38,-.43,.20]],device=device),target)
     g['ctx'].render();camera.update(1/20,force_recompute=True);im=Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:,:,:3].astype(np.uint8));d=ImageDraw.Draw(im);d.rectangle((0,0,800,54),fill=(18,25,38));d.text((12,8),f'{label} | new perturbation case 0 | {(t+1)/60:.2f}s | push 18-18.5s',fill='white');d.text((12,30),f'hold {float(env.hold[0]):.2f}s | force {float(env.contact[0].norm(dim=-1).max()):.2f}N | falls {int(env.episode_falls[0])}',fill='white');writer.append_data(np.asarray(im))
  if writer:writer.close()
  np.savez_compressed(out/(label+'_trace.npz'),**{k:np.stack([x[k] for x in traces]) for k in traces[0]})
  passed=(env.hold>=3)&(env.episode_falls==0)
  row=dict(label=label,actor=actor,normalization=norm,frames=checkpoints[actor]['frames'],seed=1062601,scale=scale,push=push,num_envs=N,successes=int(passed.sum()),ever_held_3s=int((env.best>=3).sum()),passed=passed.cpu().tolist(),final_hold=env.hold.cpu().tolist(),cases=env.cases.tolist(),diagnostics=env.diagnostics(),duration_s=time.monotonic()-start)
  save_json(out/(label+'.json'),row);summary.append(row);save_json(out/'summary.json',summary);print('REVIEW',label,row['successes'],'/',N,flush=True)
 save_json(out/'completed.json',dict(status='completed',tests=len(summary),checkpoint_hashes={k:hashlib.sha256((Path(args.checkpoint_dir)/(k+'.pth')).read_bytes()).hexdigest() for k in checkpoints}))
