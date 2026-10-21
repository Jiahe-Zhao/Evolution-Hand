"""Collision-aware opposition planning; original hand, limits, and topology filters."""
import json,sys,time
from pathlib import Path
import numpy as np,fcl
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plan_grasp import Model
from fixed_clusters import exclusions
m=Model(sys.argv[1]);prior=json.loads(Path(sys.argv[2]).read_text())[0];n=len(m.names)
excluded=set(tuple(sorted(p)) for p in exclusions(m.path));pairs=[(i,j) for i,(a,_,_) in enumerate(m.meshes) for j,(b,_,_) in enumerate(m.meshes) if j>i and a!=b and tuple(sorted((a,b))) not in excluded]
objects=[fcl.CollisionObject(shape) for _,_,shape in m.meshes];tool=fcl.CollisionObject(fcl.Box(.019,.019,.181));req=fcl.CollisionRequest(enable_contact=True,num_max_contacts=3)
# Bounding spheres only skip impossible intersections; exact convex contacts produce penalties.
bounds=[]
for (name,part,shape),(sname,points) in zip(m.meshes,m.samples):
 c=(points.min(0)+points.max(0))/2;bounds.append((c,np.linalg.norm(points-c,axis=1).max()))
lo=np.r_[np.array(m.lo)+.06,[-.13,-.06,.30,-np.pi,-np.pi,-np.pi]];hi=np.r_[np.array(m.hi)-.06,[.05,.10,.46,np.pi,np.pi,np.pi]]
x0=np.r_[[prior['q'][name] for name in m.names],prior['center'],Rotation.from_matrix(prior['rotation']).as_rotvec()];x0=np.clip(x0,lo+1e-6,hi-1e-6)
sign=np.array([1,-1,-1,-1,-1]);calls=0

def depth(a,b):
 res=fcl.CollisionResult();fcl.collide(a,b,req,res)
 return max((c.penetration_depth for c in res.contacts),default=0.) if res.is_collision else 0.

def fun(x):
 global calls
 calls+=1;q=x[:n];c=x[n:n+3];rot=Rotation.from_rotvec(x[-3:]).as_matrix();a,b,z=rot.T;p,normal=m.pads(q);rel=p-c;ts=m.fk(q);centers=[]
 for ob,(name,_,_),bound in zip(objects,m.meshes,bounds):
  T=ts[name];ob.setTransform(fcl.Transform(T[:3,:3],T[:3,3]));centers.append(T[:3,:3]@bound[0]+T[:3,3])
 centers=np.array(centers);tool.setTransform(fcl.Transform(rot,c));selfres=np.zeros(len(pairs));objres=np.zeros(len(objects))
 for k,(i,j) in enumerate(pairs):
  if np.linalg.norm(centers[i]-centers[j])<=bounds[i][1]+bounds[j][1]:selfres[k]=depth(objects[i],objects[j])
 for i,ob in enumerate(objects):
  if np.linalg.norm(centers[i]-c)<=bounds[i][1]+.092:objres[i]=depth(ob,tool)
 residual=np.r_[((rel@a)-sign*.0115)*250,(rel@b)*180,(normal+sign[:,None]*a).ravel()*5.0,((p[0]-p[1:].mean(0))@z)*300,np.maximum(abs(rel@z)-.07,0)*200,((p[1:]@z)-(p[1:]@z).mean()-np.array([.030,.010,-.010,-.030]))*180,m.clearance_residual(q,c,rot),selfres*4000,objres*4000]
 if calls%600==0:print('EVAL',calls,'norm',np.linalg.norm(residual),'selfmax',selfres.max(),'objmax',objres.max(),flush=True)
 return residual
results=[]
for seed in range(2):
 x=x0.copy()
 if seed:x[:n]=np.clip(x[:n]+np.random.default_rng(17).normal(0,.08,n),lo[:n]+1e-6,hi[:n]-1e-6)
 fit=least_squares(fun,x,bounds=(lo,hi),max_nfev=180,diff_step=1e-4,ftol=1e-7);q=fit.x[:n];c=fit.x[n:n+3];rot=Rotation.from_rotvec(fit.x[-3:]).as_matrix();p,normal=m.pads(q);hits=m.collisions(q,c,rot)
 row={'pad_face':m.face,'cost':float(np.linalg.norm(fit.fun)),'q':dict(zip(m.names,q.tolist())),'center':c.tolist(),'rotation':rot.tolist(),'pad_points':p.tolist(),'pad_normals':normal.tolist(),'normal_alignment':(-sign*(normal@rot[:,0])).tolist(),'thumb_center_offset_mm':float(((p[0]-p[1:].mean(0))@rot[:,2])*1000),'intersections':hits}
 results.append(row);Path(sys.argv[3]).write_text(json.dumps(sorted(results,key=lambda r:r['cost']),indent=2));print('RESULT',seed,row['cost'],hits,flush=True)
