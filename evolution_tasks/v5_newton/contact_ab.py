"""Paired physical contact / clearance fixture; no edits to rewards or sensors."""
import torch
from frame_math import rotate_xyzw,inverse_rotate_xyzw

def run(env):
    assert env.num_envs==4
    name={"grasp":"Object","forage":"leaf_one","strike":"Cone","branch":"branch"}[env.task]
    env.reset()
    ids=torch.arange(4,device=env.device)
    # Reset writes joint state. Propagate FK with one neutral physics step before using fingertip pose.
    env.scene.write_data_to_sim();env.ctx.step(render=False);env.scene.update(env.physics_dt)
    tip=env.hand.data.body_link_pose_w.torch[:,env.fingertip_grid.tipids[0]].clone()
    target=env.pose(name).clone()
    offset=env.tensor([[-.014,0,.008],[-.080,0,.008],[-.013,0,.008],[-.080,0,.008]])
    if env.task=="branch":
        # End-on approach keeps the long cylinder shaft away from adjacent digits.
        offset[:,0]=env.tensor([-.093,-.16,-.093,-.16])
    if env.task=="forage":offset[:,0]-=.041
    if env.config.get("left_pad"):offset[:,0]*=-1
    target[:,:3]=tip[:,:3]+rotate_xyzw(tip[:,3:],offset);target[:,3:]=tip[:,3:]
    if env.task=="branch":
        from scipy.spatial.transform import Rotation
        q=(Rotation.from_quat(tip[:,3:].cpu().numpy())*Rotation.from_euler("y",90,degrees=True)).as_quat()
        target[:,3:]=env.tensor(q)
    env.put(name,target,ids)
    a=torch.zeros(4,env.action_size,device=env.device)
    lim=env.limits[env.fids];a[:,:len(lim)]=2*(env.target[:,env.fids]-lim[:,0])/(lim[:,1]-lim[:,0])-1
    points=[]
    for step in range(8):
        env.contact_trace=[];env.filter_trace=[]
        _,rew,t,tr,info=env.step(a)
        tip_now=env.hand.data.body_link_pose_w.torch[:,env.fingertip_grid.tipids[0]]
        rel=inverse_rotate_xyzw(tip_now[:,3:],env.pose(name)[:,:3]-tip_now[:,:3])
        points.append(dict(step=step,object_in_tip=rel.tolist(),raw_substeps=env.contact_trace,filter_trace=env.filter_trace,force=env.fingertip_grid.total_force.sum(1).tolist(),reward=rew.tolist(),success=info['episode_success'].tolist(),done=(t|tr).tolist()))
        if (t|tr).any():break
    env.contact_trace=None
    return points
