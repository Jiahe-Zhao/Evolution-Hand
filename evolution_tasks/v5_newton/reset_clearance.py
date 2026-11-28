"""Convex URDF collision checks for reset-only placement, preserving rollout freedom."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh,fcl
from scipy.spatial.transform import Rotation

class Clearance:
    def __init__(self,urdf,hand,asset=None):
        self.hand=hand;self.parts=[];self.cache={}
        if asset:
            from pxr import Usd,UsdGeom,UsdPhysics,Gf
            stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache()
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                if prim.GetAttribute('physics:collisionEnabled').Get() is False:continue
                body=prim.GetParent()
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
                if not body or body.GetName() not in hand.body_names:raise ValueError('Collision body missing')
                mat=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
                if prim.IsA(UsdGeom.Mesh):local=UsdGeom.Mesh(prim).GetPointsAttr().Get()
                elif prim.IsA(UsdGeom.Sphere):
                    # Circumscribed hull: do not miss the ellipsoid surface between sampled vertices.
                    sphere=trimesh.creation.icosphere(subdivisions=2,radius=1.)
                    distance=np.abs((sphere.face_normals*sphere.triangles_center).sum(1)).min()
                    local=sphere.vertices*(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())/distance)
                elif prim.IsA(UsdGeom.Cube):local=trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())]*3).vertices
                else:raise ValueError('Unsupported USD collision shape: '+str(prim.GetPath()))
                pts=np.array([mat.Transform(Gf.Vec3d(*map(float,v))) for v in local])
                hull=trimesh.convex.convex_hull(pts)
                faces=np.column_stack([np.full(len(hull.faces),3),hull.faces]).astype(np.int32).ravel()
                self.parts.append((hand.body_names.index(body.GetName()),fcl.Convex(np.asarray(hull.vertices),len(hull.faces),faces)))
            if not self.parts:raise ValueError('Empty USD collision audit')
            return
        path=Path(urdf)
        for link in ET.parse(path).getroot().findall('link'):
            for collision in link.findall('collision'):
                node=collision.find('geometry/mesh')
                if node is None:raise ValueError('Unsupported reset collision geometry')
                mesh=trimesh.load(path.parent/node.get('filename'),force='mesh')
                mesh.apply_scale(np.fromstring(node.get('scale','1 1 1'),sep=' '))
                origin=collision.find('origin');tf=np.eye(4)
                if origin is not None:
                    tf[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                    tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                mesh.apply_transform(tf);hull=mesh.convex_hull
                faces=np.column_stack([np.full(len(hull.faces),3),hull.faces]).astype(np.int32).ravel()
                shape=fcl.Convex(np.asarray(hull.vertices),len(hull.faces),faces)
                self.parts.append((hand.body_names.index(link.get('name')),shape))
    def objects(self,index):
        poses=self.hand.data.body_link_pose_w.torch[index].detach().cpu().numpy()
        return [fcl.CollisionObject(shape,fcl.Transform(Rotation.from_quat(poses[i,3:]).as_matrix(),poses[i,:3])) for i,shape in self.parts]
    @staticmethod
    def collides(objects,center,quaternion,kind,size,margin):
        shape=fcl.Cylinder(size[0]+margin,size[1]+2*margin) if kind=='cylinder' else fcl.Box(*(np.asarray(size)+2*margin))
        target=fcl.CollisionObject(shape,fcl.Transform(Rotation.from_quat(quaternion).as_matrix(),np.asarray(center)))
        for obj in objects:
            result=fcl.CollisionResult()
            fcl.collide(target,obj,fcl.CollisionRequest(),result)
            if result.is_collision:return True
        return False
    def branch_valid(self,index,pose,radius,height):
        return not self.collides(self.objects(index),pose[:3],pose[3:],'cylinder',[radius,height],.0005)
    def strike_offset(self,index,pose,size):
        q=self.hand.data.joint_pos.torch[index].detach().cpu().numpy()
        root=self.hand.data.body_link_pose_w.torch[index,0,:3].detach().cpu().numpy()
        key=tuple(np.round(np.concatenate([q,pose[:3]-root,pose[3:]]),6))
        if key in self.cache:return self.cache[key]
        objects=self.objects(index)
        offsets=sorted(((x*.001,y*.001,0.) for x in range(-30,31) for y in range(-30,31)),key=lambda v:v[0]**2+v[1]**2)
        for offset in offsets:
            if not self.collides(objects,pose[:3]+offset,pose[3:],'box',size,.001):
                self.cache[key]=offset;return offset
        raise RuntimeError('No clear Strike reset within 30mm; reject scene, not morphology fitness')
