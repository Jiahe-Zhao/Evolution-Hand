"""Cache immutable imported convex collision shapes; update only body poses."""
import numpy as np
import fcl,trimesh
from scipy.spatial.transform import Rotation

class CollisionCache:
 def __init__(self,e):
  from imported_hand_geometry import hand_collision_objects
  _,metadata=hand_collision_objects(e,0)
  pos=e.hand.data.body_pos_w[0].detach().cpu().numpy();q=e.hand.data.body_quat_w[0].detach().cpu().numpy()
  self.items=[]
  for item in metadata:
   if 'vertices' not in item:raise RuntimeError('Unsupported collision geometry '+str(item))
   i=e.hand.body_names.index(item['body']);r=Rotation.from_quat(q[i,[1,2,3,0]]).as_matrix()
   vertices=(np.asarray(item['vertices'])-pos[i])@r
   hull=trimesh.convex.convex_hull(vertices)
   faces=np.column_stack([np.full(len(hull.faces),3),hull.faces]).astype(np.int32).ravel()
   shape=fcl.Convex(np.asarray(hull.vertices),len(hull.faces),faces)
   self.items.append((item['path'],i,shape))
  self.e=e
 def collisions(self,index,center,quat,margin=.0005):
  e=self.e;p=e.hand.data.body_pos_w[index].detach().cpu().numpy();q=e.hand.data.body_quat_w[index].detach().cpu().numpy()
  radius=float(e.cfg.branch_cfg.spawn.radius)+margin;height=float(e.cfg.branch_cfg.spawn.height)+2*margin
  branch=fcl.CollisionObject(fcl.Cylinder(radius,height),fcl.Transform(Rotation.from_quat(np.asarray(quat)[[1,2,3,0]]).as_matrix(),np.asarray(center)))
  hits=[]
  for name,i,shape in self.items:
   body=fcl.CollisionObject(shape,fcl.Transform(Rotation.from_quat(q[i,[1,2,3,0]]).as_matrix(),p[i]))
   result=fcl.CollisionResult();fcl.collide(branch,body,fcl.CollisionRequest(),result)
   if result.is_collision:hits.append(name)
  return hits

def anatomical_frame(e):
 names=e.hand.body_names;p=e.hand.data.body_pos_w[0].detach().cpu().numpy()-e.scene.env_origins[0].cpu().numpy()
 if e.variant=='shadow':palm='robot0_palm';base=['robot0_'+x+'proximal' for x in ['ff','mf','rf','lf']]
 else:palm='link_palm_chassis' if 'link_palm_chassis' in names else 'link_0_0';base=['link_'+str(i)+'_0' for i in range(2,6)]
 points=p[[names.index(n) for n in base]];origin=points.mean(0)
 x=points[0]-points[-1];x/=np.linalg.norm(x)
 y=origin-p[names.index(palm)];y-=np.dot(y,x)*x;y/=np.linalg.norm(y)
 z=np.cross(x,y);z/=np.linalg.norm(z)
 return dict(origin_world=origin.tolist(),rotation=np.stack((x,y,z),axis=1).tolist(),definition='origin=mean long proximal origins; x=little to index; y=palm toward proximal group, orthogonal to x; z=x cross y')

def verify_cache(e,cache):
 """Compare cached and original convex shapes at several probe positions."""
 from imported_hand_geometry import hand_collision_objects
 objects,_=hand_collision_objects(e,0)
 frame=anatomical_frame(e);origin=np.asarray(frame['origin_world'])+e.scene.env_origins[0].cpu().numpy()
 rot=np.asarray(frame['rotation']);quat=Rotation.from_matrix(rot@Rotation.from_euler('y',90,degrees=True).as_matrix()).as_quat()[[3,0,1,2]]
 comparisons=[]
 for offset in ([0,0,0],[0,.03,.03],[0,-.03,.03],[0,.03,-.03],[0,0,.15]):
  center=origin+rot@np.asarray(offset)
  branch=fcl.CollisionObject(fcl.Cylinder(float(e.cfg.branch_cfg.spawn.radius)+.0005,float(e.cfg.branch_cfg.spawn.height)+.001),fcl.Transform(Rotation.from_quat(quat[[1,2,3,0]]).as_matrix(),center))
  reference=[]
  for name,body in objects:
   r=fcl.CollisionResult();fcl.collide(branch,body,fcl.CollisionRequest(),r)
   if r.is_collision:reference.append(name)
  cached=cache.collisions(0,center,quat)
  if sorted(reference)!=sorted(cached):raise RuntimeError('Cached collision geometry differs from reference')
  comparisons.append(dict(offset=offset,collision_count=len(cached),matches=True))
 return comparisons

def export_kinematics(e):
 from pxr import Usd,UsdPhysics
 from scipy.spatial.transform import Rotation
 def vector(v):return [float(x) for x in v]
 def matrix(pos,quat):
  m=np.eye(4);m[:3,:3]=Rotation.from_quat([*vector(quat.GetImaginary()),float(quat.GetReal())]).as_matrix();m[:3,3]=vector(pos);return m.tolist()
 joints=[]
 for p in Usd.PrimRange(e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
  if not p.IsA(UsdPhysics.Joint):continue
  j=UsdPhysics.Joint(p);parents=j.GetBody0Rel().GetTargets();children=j.GetBody1Rel().GetTargets()
  if not parents or not children:continue
  kind=p.GetTypeName()
  if kind not in ['PhysicsRevoluteJoint','PhysicsFixedJoint']:raise RuntimeError('Unsupported FK joint '+kind)
  joints.append(dict(name=p.GetName(),parent=parents[0].name,child=children[0].name,kind=kind,axis=UsdPhysics.RevoluteJoint(p).GetAxisAttr().Get() if kind=='PhysicsRevoluteJoint' else None,frame0=matrix(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),frame1=matrix(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get()),tendon_attributes={a.GetName():str(a.Get()) for a in p.GetAttributes() if 'tendon' in a.GetName().lower()}))
 from imported_hand_geometry import hand_collision_objects
 _,geometry=hand_collision_objects(e,0)
 positions=e.hand.data.body_pos_w[0].cpu().numpy();quats=e.hand.data.body_quat_w[0].cpu().numpy()
 tips=[]
 for index in e.tip_ids:
  name=e.hand.body_names[index];items=[g for g in geometry if g.get('body')==name]
  pads=[g for g in items if 'finger_pad' in g['path']];items=pads or items
  points=np.concatenate([g['vertices'] for g in items]);r=Rotation.from_quat(quats[index,[1,2,3,0]]).as_matrix();local=(points-positions[index])@r
  center=(local.max(0)+local.min(0))/2;half=(local.max(0)-local.min(0))/2
  tips.append(dict(body=name,center=center.tolist(),radius=float(np.sort(half)[1])))
 return dict(joints=joints,tips=tips,body_names=e.hand.body_names,body_pos=positions.tolist(),body_quat=quats.tolist(),joint_names=e.hand.joint_names,q=e.hand.data.joint_pos[0].tolist(),env_origin=e.scene.env_origins[0].tolist())
