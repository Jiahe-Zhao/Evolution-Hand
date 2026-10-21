"""Geometric contact candidates, not measured force allocation per mesh."""
def audit(e,out,step):
 import json,numpy as np,fcl
 from scipy.spatial.transform import Rotation
 from imported_hand_geometry import hand_collision_objects
 state=e.cone.data.root_state_w[0].cpu().numpy();obj=fcl.CollisionObject(fcl.Box(*e.cfg.Cone_cfg.spawn.size),fcl.Transform(Rotation.from_quat(state[[4,5,6,3]]).as_matrix(),state[:3]));bodies,_=hand_collision_objects(e,0);rows=[]
 for path,body in bodies:
  r=fcl.CollisionResult();fcl.collide(obj,body,fcl.CollisionRequest(enable_contact=True,num_max_contacts=8),r)
  distance=fcl.distance(obj,body,fcl.DistanceRequest(),fcl.DistanceResult())
  rows.append({'path':path,'distance_m':float(distance),'intersecting':bool(r.is_collision),'contact_positions':[c.pos.tolist() for c in r.contacts],'depth_m':max((float(c.penetration_depth) for c in r.contacts),default=0.)})
 (out/f'contact_geometry_{step}.json').write_text(json.dumps({'step':step,'note':'geometric candidates; finger-level sensor forces do not identify individual collision mesh','shapes':sorted(rows,key=lambda r:r['distance_m'])},indent=2))
