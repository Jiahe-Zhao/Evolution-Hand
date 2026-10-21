"""Audit imported collision geometry against the reset tool, in physical body frames."""
import json
from pathlib import Path
import numpy as np
import trimesh
import fcl
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


def hand_collision_objects(env, env_index=0):
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
    if not objects:raise RuntimeError("No imported collision geometry")
    return objects, geometry
