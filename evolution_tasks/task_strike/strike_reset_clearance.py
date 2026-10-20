"""Audit imported collision geometry against the reset tool, in physical body frames."""
import json
from pathlib import Path
import numpy as np
import trimesh
import fcl
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


def audit(env, output=None, env_index=0):
    stage=env.sim.stage
    cache=UsdGeom.XformCache()
    names=env.hand.body_names
    positions=env.hand.data.body_pos_w[env_index].detach().cpu().numpy()
    quats=env.hand.data.body_quat_w[env_index].detach().cpu().numpy()
    objects=[];geometry=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in ('Mesh','Cube','Capsule','Cylinder','Sphere'):continue
        collision_owner=prim
        while collision_owner.IsValid() and not collision_owner.HasAPI(UsdPhysics.CollisionAPI):collision_owner=collision_owner.GetParent()
        if not collision_owner.IsValid():continue
        if UsdPhysics.CollisionAPI(collision_owner).GetCollisionEnabledAttr().Get() is False:continue
        body=prim
        while body.IsValid() and body.GetName() not in names:body=body.GetParent()
        if not body.IsValid():continue
        index=names.index(body.GetName())
        kind=prim.GetTypeName();approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
        if kind=='Mesh':
            points=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
        elif kind=='Cube':
            size=UsdGeom.Cube(prim).GetSizeAttr().Get();points=trimesh.creation.box(extents=[size]*3).vertices
        elif kind in ('Capsule','Cylinder','Sphere'):
            schema=getattr(UsdGeom,kind)(prim);radius=schema.GetRadiusAttr().Get()
            if kind=='Sphere':mesh=trimesh.creation.icosphere(subdivisions=3,radius=radius)
            elif kind=='Cylinder':mesh=trimesh.creation.cylinder(radius=radius,height=schema.GetHeightAttr().Get(),sections=64)
            else:mesh=trimesh.creation.capsule(radius=radius,height=schema.GetHeightAttr().Get(),count=[32,32])
            points=mesh.vertices-mesh.bounds.mean(axis=0)
            if kind!='Sphere':
                axis=str(schema.GetAxisAttr().Get())
                if axis=='X':points=points[:,[2,1,0]]
                if axis=='Y':points=points[:,[0,2,1]]
        else:
            geometry.append(dict(path=str(prim.GetPath()),unsupported=kind));continue
        relative=np.array(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()).T
        points=trimesh.transform_points(points,relative)
        rotation=Rotation.from_quat(quats[index][[1,2,3,0]]).as_matrix()
        points=points@rotation.T+positions[index]
        hull=trimesh.convex.convex_hull(points)
        faces=np.column_stack([np.full(len(hull.faces),3),hull.faces]).astype(np.int32).ravel()
        shape=fcl.Convex(np.asarray(hull.vertices),len(hull.faces),faces)
        obj=fcl.CollisionObject(shape)
        path=str(prim.GetPath());objects.append((path,obj))
        geometry.append(dict(path=path,body=body.GetName(),kind=kind,approximation=approx,vertex_count=len(points),world_bounds=hull.bounds.tolist(),vertices=points.tolist()))
    pos=env.cone.data.root_pos_w[env_index].detach().cpu().numpy()
    quat=env.cone.data.root_quat_w[env_index].detach().cpu().numpy()
    rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    size=np.array(env.cfg.Cone_cfg.spawn.size)
    def collisions(center, margin=0., details=False):
        tool=fcl.CollisionObject(fcl.Box(*(size+2*margin)),fcl.Transform(rotation,center))
        hits=[]
        for name,obj in objects:
            result=fcl.CollisionResult();fcl.collide(tool,obj,fcl.CollisionRequest(num_max_contacts=16,enable_contact=details),result)
            if result.is_collision:
                if not details:return True
                hits.append(dict(path=name,depth_m=max((float(c.penetration_depth) for c in result.contacts),default=0.)))
        return hits if details else False
    if not objects:raise RuntimeError('No imported collision geometry; cannot certify clearance')
    initial=collisions(pos,details=True)
    safe=None
    if initial or collisions(pos, margin=.001):
        # Nearest horizontal collision-free placement within 30 mm; preserve height.
        offsets=sorted(((x*.001,y*.001,0.) for x in range(-30,31) for y in range(-30,31)),key=lambda p:p[0]**2+p[1]**2)
        for delta in offsets:
            if not collisions(pos+delta,margin=.001):safe=list(delta);break
    else:safe=[0.,0.,0.]
    report=dict(geometry=geometry,initial_collisions=initial,safe_offset_m=safe,query_margin_m=.001,
                note='Convex hull of imported collision mesh; exact for convexHull, conservative for other approximations.')
    if output is not None:
        Path(output,'imported_collision_audit.json').write_text(json.dumps(report,indent=2))
    env.strike_last_clearance_report = {key:value for key,value in report.items() if key!='geometry'}
    return safe


def correct_reset_tool_placement(env, env_ids):
    """Cache certified offsets by reset pose, shared by identical cloned poses.

    Geometry remains unchanged. Every distinct reset pose is checked, including
    partial resets; a failed search rejects the scene instead of spawning inside it.
    """
    import torch
    ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if ids.numel()==0:return
    cache=getattr(env,'_strike_clearance_cache',None)
    if cache is None:
        cache={};env._strike_clearance_cache=cache
    states=env.cone.data.root_state_w[ids].clone()
    local_positions=states[:,:3]-env.scene.env_origins[ids]
    signatures=torch.cat((env.hand.data.joint_pos[ids],local_positions,states[:,3:7]),dim=1).detach().cpu().numpy()
    groups={}
    for row,signature in enumerate(signatures):
        # Sub-micrometre grouping tolerance is much smaller than the 1 mm clearance.
        key=tuple(np.round(signature,6));groups.setdefault(key,[]).append(row)
    for key,rows in groups.items():
        if key not in cache:
            offset=audit(env,env_index=int(ids[rows[0]]))
            if offset is None:raise RuntimeError('Strike reset: no collision-free tool placement within 30 mm; inspect morphology before training')
            cache[key]=offset
        states[rows,:3]+=torch.tensor(cache[key],device=env.device,dtype=states.dtype)
    states[:,7:]=0.
    env.cone.write_root_state_to_sim(states,env_ids=ids)
