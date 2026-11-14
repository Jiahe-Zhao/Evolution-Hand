"""Paired physical contact / clearance fixture; no edits to rewards or sensors."""
import torch
from frame_math import rotate_xyzw

def run(env):
    assert env.num_envs==4
    name={"grasp":"Object","forage":"leaf_one","strike":"Cone","branch":"branch"}[env.task]
    env.reset()
    ids=torch.arange(4,device=env.device)
    tip=env.hand.data.body_link_pose_w.torch[:,env.fingertip_grid.tipids[0]].clone()
    target=env.pose(name).clone()
    offset=env.tensor([[-.014,0,.008],[-.080,0,.008],[-.013,0,.008],[-.080,0,.008]])
    if env.task=="forage":offset[:,0]-=.041
    if env.config.get("left_pad"):offset[:,0]*=-1
    target[:,:3]=tip[:,:3]+rotate_xyzw(tip[:,3:],offset);target[:,3:]=tip[:,3:]
    env.put(name,target,ids)
    a=torch.zeros(4,env.action_size,device=env.device)
    lim=env.limits[env.fids];a[:,:len(lim)]=2*(env.target[:,env.fids]-lim[:,0])/(lim[:,1]-lim[:,0])-1
    points=[]
    for step in range(8):
        _,rew,t,tr,info=env.step(a)
        points.append(dict(step=step,force=env.fingertip_grid.total_force.sum(1).tolist(),reward=rew.tolist(),success=info['episode_success'].tolist(),done=(t|tr).tolist()))
        if (t|tr).any():break
    return points
