"""Plan pad-side opposition with an unchanged URDF and original joint limits."""
import json,sys,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import trimesh,fcl

class Model:
 def __init__(self,path):
  self.path=Path(path);r=ET.parse(path).getroot();self.links={l.get('name'):l for l in r.findall('link')};self.joints=[];self.names=[];self.lo=[];self.hi=[]
  for j in r.findall('joint'):
   o=j.find('origin');T=np.eye(4)
   if o is not None:T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
   idx=None;axis=np.zeros(3)
   if j.get('type')=='revolute':
    idx=len(self.names);self.names.append(j.get('name'));lim=j.find('limit');self.lo.append(float(lim.get('lower')));self.hi.append(float(lim.get('upper')));axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
   self.joints.append((j.find('parent').get('link'),j.find('child').get('link'),T,axis,idx))
  self.meshes=[];self.pad_specs=[];self.samples=[]
  for name,l in self.links.items():
   for c in l.findall('collision'):
    m=c.find('geometry/mesh')
    if m is None:continue
    mesh=trimesh.load(self.path.parent/m.get('filename'),force='mesh');T=np.eye(4);o=c.find('origin')
    if o is not None:T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix();T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
    mesh.apply_transform(T);self.samples.append((name,np.vstack([mesh.vertices,mesh.triangles_center])));h=mesh.convex_hull;faces=np.column_stack([np.full(len(h.faces),3),h.faces]).astype(np.int32).ravel();self.meshes.append((name,c.get('name'),fcl.Convex(np.asarray(h.vertices),len(h.faces),faces)))
    if c.get('name')=='finger_pad':
     bounds=mesh.bounds;p=bounds.mean(0);finger=int(name.split('_')[1]);normal=np.array([-1.,0.,0.]) if finger==1 else np.array([1.,0.,0.]);p[0]=bounds[0 if finger==1 else 1,0];self.pad_specs.append((finger,name,p,normal))
  self.pad_specs.sort();self.root=np.eye(4);self.root[:3,:3]=Rotation.from_euler('y',np.pi).as_matrix();self.root[:3,3]=[-.027,.07,.38]
 def fk(self,q):
  ts={'link_0_0':self.root};todo=list(self.joints)
  while todo:
   rest=[]
   for parent,child,T,axis,i in todo:
    if parent not in ts:rest.append((parent,child,T,axis,i));continue
    J=np.eye(4)
    if i is not None:J[:3,:3]=Rotation.from_rotvec(axis*q[i]).as_matrix()
    ts[child]=ts[parent]@T@J
   if len(rest)==len(todo):raise ValueError('Disconnected kinematic tree')
   todo=rest
  return ts
 def clearance_residual(self,q,c,rot):
  ts=self.fk(q);rows=[]
  for name,points in self.samples:
   T=ts[name];pts=(points@T[:3,:3].T+T[:3,3]-c)@rot
   depth=np.min(np.array([.0095,.0095,.0905])-np.abs(pts),axis=1)
   rows.append(max(0.,float(depth.max())))
  return np.array(rows)*700
 def pads(self,q):
  ts=self.fk(q);p=[];n=[]
  for _,name,local,normal in self.pad_specs:p.append(ts[name][:3,:3]@local+ts[name][:3,3]);n.append(ts[name][:3,:3]@normal)
  return np.array(p),np.array(n)
 def collisions(self,q,center,rotation,size=(.018,.018,.18)):
  ts=self.fk(q);obj=fcl.CollisionObject(fcl.Box(*size),fcl.Transform(rotation,center));hits=[]
  for name,part,shape in self.meshes:
   T=ts[name];r=fcl.CollisionResult();fcl.collide(obj,fcl.CollisionObject(shape,fcl.Transform(T[:3,:3],T[:3,3])),fcl.CollisionRequest(enable_contact=True,num_max_contacts=8),r)
   if r.is_collision:hits.append({'link':name,'part':part,'depth_m':max((c.penetration_depth for c in r.contacts),default=0.)})
  return hits

def plan(path,out):
 m=Model(path);n=len(m.names);sign=np.array([1,-1,-1,-1,-1]);rng=np.random.default_rng(31);lo=np.r_[m.lo,[-.11,-.04,.32,-np.pi,-np.pi,-np.pi]];hi=np.r_[m.hi,[.025,.06,.44,np.pi,np.pi,np.pi]];results=[]
 def fun(x):
  p,normal=m.pads(x[:n]);c=x[n:n+3];rot=Rotation.from_rotvec(x[-3:]).as_matrix();a,b,z=rot.T;rel=p-c
  return np.r_[((rel@a)-sign*.011)*250,(rel@b)*180,(normal+sign[:,None]*a).ravel()*1.5,((p[0]-p[1:].mean(0))@z)*300,np.maximum(abs(rel@z)-.07,0)*200,((p[1:]@z)-(p[1:]@z).mean()-np.array([.024,.008,-.008,-.024]))*220,m.clearance_residual(x[:n],c,rot)]
 for k in range(4):
  x=(lo+hi)/2;x[:n]=np.array(m.lo)+rng.uniform(.15,.85,n)*(np.array(m.hi)-m.lo);x[n:]=[-.065,.005,.37,*rng.uniform(-2,2,3)]
  if k==0:
   prior=json.loads(Path('/home/zjh/Evolution_PC/outputs/contact_control_calibration_20261021/plans.json').read_text())[0]
   x[:n]=[prior['q'][name] for name in m.names];x[n:n+3]=prior['center'];x[-3:]=Rotation.from_matrix(prior['rotation']).as_rotvec()
  fit=least_squares(fun,x,bounds=(lo+1e-6,hi-1e-6),max_nfev=180,ftol=1e-7);q=fit.x[:n];p,normal=m.pads(q);c=fit.x[n:n+3];rot=Rotation.from_rotvec(fit.x[-3:]).as_matrix();a,b,z=rot.T;hits=m.collisions(q,c,rot)
  row={'cost':float(np.linalg.norm(fit.fun)),'q':dict(zip(m.names,q.tolist())),'center':c.tolist(),'rotation':rot.tolist(),'pad_points':p.tolist(),'pad_normals':normal.tolist(),'pad_faces_local':[[-1,0,0]]+[[1,0,0]]*4,'normal_alignment':(-sign*(normal@a)).tolist(),'thumb_center_offset_mm':float(((p[0]-p[1:].mean(0))@z)*1000),'intersections':hits}
  results.append(row);print(k,round(row['cost'],3),round(row['thumb_center_offset_mm'],3),len(hits),flush=True)
  Path(out).write_text(json.dumps(sorted(results,key=lambda r:(bool(r['intersections']),r['cost'])),indent=2))
if __name__=='__main__':plan(*sys.argv[1:])
