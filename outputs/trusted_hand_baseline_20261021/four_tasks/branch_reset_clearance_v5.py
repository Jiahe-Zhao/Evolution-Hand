"""Collision-free branch placement using imported geometry, including fingertips."""
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
from imported_hand_geometry import hand_collision_objects


def correct(env, env_ids, margin=.001, half_extent=.016, grid=.001):
    import torch
    if env_ids is None:env_ids=range(env.num_envs)
    ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if not ids.numel():return
    states=env.branch.data.root_state_w[ids].clone()
    radius=float(env.cfg.branch_cfg.spawn.radius);height=float(env.cfg.branch_cfg.spawn.height)
    count=int(round(half_extent/grid))
    # Search nearest displacement first in world xyz, independent of hand axes.
    offsets=sorted(((x*grid,y*grid,z*grid) for x in range(-count,count+1)
                    for y in range(-count,count+1) for z in range(-count,count+1)
                    if x*x+y*y+z*z<=count*count),key=lambda x:sum(v*v for v in x))
    reports=[]
    for row,index in enumerate(ids.tolist()):
        objects,_=hand_collision_objects(env,index)
        state=states[row].detach().cpu().numpy()
        center=state[:3];rotation=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
        def hits(pos,padding=0.,details=False):
            shape=fcl.Cylinder(radius+padding,height+2*padding)
            branch=fcl.CollisionObject(shape,fcl.Transform(rotation,pos))
            result_rows=[]
            for name,body in objects:
                result=fcl.CollisionResult()
                fcl.collide(branch,body,fcl.CollisionRequest(num_max_contacts=16,enable_contact=details),result)
                if result.is_collision:
                    if not details:return True
                    result_rows.append(dict(path=name,depth_m=max((float(c.penetration_depth) for c in result.contacts),default=0.)))
            return result_rows if details else False
        initial=hits(center,details=True)
        selected=None
        for offset in offsets:
            if not hits(center+offset,margin):
                selected=np.asarray(offset);break
        if selected is None:
            raise RuntimeError(f'No collision-free branch reset within {half_extent} m for env {index}')
        states[row,:3]+=torch.tensor(selected,device=env.device,dtype=states.dtype)
        reports.append(dict(env_index=index,initial_collisions=initial,offset_m=selected.tolist(),margin_m=margin))
    env.branch.write_root_state_to_sim(states,ids)
    env.branch_clearance_report=reports
