def audit(e,task,out):
 import json,fcl,numpy as np
 from scipy.spatial.transform import Rotation
 from imported_hand_geometry import hand_collision_objects
 objects,_=hand_collision_objects(e,0)
 names={'grasp':['grasp_object'],'branch':['branch'],'forage':['food','leaf_one','leaf_two'],'strike':['cone']}[task]
 rows=[]
 for name in names:
  obj=getattr(e,name);cfg=obj.cfg.spawn;state=obj.data.root_state_w[0].cpu().numpy()
  if task=='grasp':shape=fcl.Sphere(cfg.radius)
  elif task=='branch':shape=fcl.Cylinder(cfg.radius,cfg.height)
  else:shape=fcl.Box(*cfg.size)
  target=fcl.CollisionObject(shape,fcl.Transform(Rotation.from_quat(state[[4,5,6,3]]).as_matrix(),state[:3]))
  for path,body in objects:
   result=fcl.CollisionResult();fcl.collide(target,body,fcl.CollisionRequest(enable_contact=True,num_max_contacts=16),result)
   if result.is_collision:rows.append({'object':name,'hand_shape':path,'depth_m':max((float(c.penetration_depth) for c in result.contacts),default=0.)})
 (out/'startup_geometry.json').write_text(json.dumps({'time':'after explicit reset, before first scripted action','intersections':rows},indent=2))
