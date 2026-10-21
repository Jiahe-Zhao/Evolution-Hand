"""Long-side grasp: flexion axes follow tool length; only grasp planning changes."""
import json,sys,time
from pathlib import Path
import numpy as np,fcl
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plan_grasp import Model
from fixed_clusters import exclusions
m=Model(sys.argv[1],face=sys.argv[4]);prior=json.loads(Path(sys.argv[2]).read_text())[0];n=len(m.names)
excluded=set(tuple(sorted(p)) for p in exclusions(m.path));pairs=[(i,j) for i,(a,_,_) in enumerate(m.meshes) for j,(b,_,_) in enumerate(m.meshes) if j>i and a!=b and tuple(sorted((a,b))) not in excluded]
objects=[fcl.CollisionObject(shape) for _,_,shape in m.meshes];tool=fcl.CollisionObject(fcl.Box(.019,.019,.181));req=fcl.CollisionRequest(enable_contact=True,num_max_contacts=3)
# Bounding spheres only skip impossible intersections; exact convex contacts produce penalties.
bounds=[]
for (name,part,shape),(sname,points) in zip(m.meshes,m.samples):
 c=(points.min(0)+points.max(0))/2;bounds.append((c,np.linalg.norm(points-c,axis=1).max()))
lo=np.r_[np.array(m.lo)+1e-5,[-.13,-.06,.30,-np.pi,-np.pi,-np.pi]];hi=np.r_[np.array(m.hi)-1e-5,[.05,.10,.46,np.pi,np.pi,np.pi]]
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
 # Align long-finger flexion axes with tool length, allowing natural deviation.
 axes=np.array([ts[name][:3,:3]@np.array([1.,0.,0.]) for _,name,_,_ in m.pad_specs[1:]])
 curl=np.cross(axes,z).ravel()*6
 residual=np.r_[curl,((rel@a)-sign*.0105)*500,(rel@b)*180,((normal+sign[:,None]*a)*np.array([10,4,4,4,4])[:,None]).ravel(),((p[0]-p[1:].mean(0))@z)*700,np.maximum(abs(rel@z)-.07,0)*200,((p[1:]@z)-(p[1:]@z).mean()-np.array([.015,.005,-.005,-.015]))*180,m.clearance_residual(q,c,rot),selfres*4000,objres*4000]
 if calls%600==0:print('EVAL',calls,'norm',np.linalg.norm(residual),'selfmax',selfres.max(),'objmax',objres.max(),flush=True)
 return residual
results=[]
for seed in range(3):
 x=x0.copy()
 if seed:
  rng=np.random.default_rng(2100+seed)
  x[:n]=np.array(m.lo)+rng.uniform(.1,.8,n)*(np.array(m.hi)-m.lo)
  points,_=m.pads(x[:n]);x[n:n+3]=np.clip(points.mean(0),lo[n:n+3]+1e-6,hi[n:n+3]-1e-6)
  axis=m.fk(x[:n])[m.pad_specs[2][1]][:3,0];a=points[0]-points[1:].mean(0);a-=axis*np.dot(a,axis);a/=np.linalg.norm(a);b=np.cross(axis,a)
  x[-3:]=Rotation.from_matrix(np.column_stack([a,b,axis])).as_rotvec()
 fit=least_squares(fun,x,bounds=(lo,hi),max_nfev=110,diff_step=1e-4,ftol=1e-7);q=fit.x[:n];c=fit.x[n:n+3];rot=Rotation.from_rotvec(fit.x[-3:]).as_matrix();p,normal=m.pads(q);hits=m.collisions(q,c,rot)
 axes=np.array([m.fk(q)[name][:3,0] for _,name,_,_ in m.pad_specs[1:]])
 row={'pad_face':m.face,'curl_axis_deviation_deg':np.degrees(np.arccos(np.clip(np.abs(axes@rot[:,2]),0,1))).tolist(),'cost':float(np.linalg.norm(fit.fun)),'q':dict(zip(m.names,q.tolist())),'center':c.tolist(),'rotation':rot.tolist(),'pad_points':p.tolist(),'pad_normals':normal.tolist(),'normal_alignment':(-sign*(normal@rot[:,0])).tolist(),'thumb_center_offset_mm':float(((p[0]-p[1:].mean(0))@rot[:,2])*1000),'intersections':hits}
 results.append(row);Path(sys.argv[3]).write_text(json.dumps(sorted(results,key=lambda r:(bool(r['intersections']),r['cost'])),indent=2));print('RESULT',seed,row['cost'],hits,flush=True)
