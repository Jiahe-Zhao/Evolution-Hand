"""Rounded, mildly top-wide convex stone, preserving original local/world bounds and mass."""
from pathlib import Path
import numpy as np,json,hashlib
from xml.etree import ElementTree as E
from scipy.spatial import ConvexHull
from scipy.interpolate import PchipInterpolator
import mujoco as mj
P=Path(__file__).parent;A=P/'assets';A.mkdir(exist_ok=True);O=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07')
src=O/'object.xml';original=A/'object_original_v1.xml'
if not original.exists():original.write_bytes(src.read_bytes())
r=E.parse(original);mesh=r.find("asset/mesh[@name='peb0']");old=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3);lo=old.min(0);hi=old.max(0)
ang=.2;rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
# Support planes in both original local axes and initial world axes.
ns=np.vstack([np.eye(2),-np.eye(2),rot,-rot]);bounds=np.max(old[:,:2]@ns.T,axis=0)
def clip(poly,n,b):
 out=[]
 for a,c in zip(poly,np.roll(poly,-1,axis=0)):
  da=a@n-b;dc=c@n-b
  if da<=1e-12:out.append(a)
  if (da<0)!=(dc<0):out.append(a+(c-a)*da/(da-dc))
 return np.array(out)
def outline(radius):
 poly=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
 for n,b in zip(ns,bounds):poly=clip(poly,n,b-radius)
 if len(poly)<3:return None
 theta=np.unique(np.r_[np.linspace(0,2*np.pi,40,endpoint=False),np.arctan2(ns[:,1],ns[:,0])]);cir=radius*np.c_[np.cos(theta),np.sin(theta)]
 pts=(poly[:,None,:]+cir[None,:,:]).reshape(-1,2);pts=pts[ConvexHull(pts).vertices]
 return pts
chosen=None
for rad in np.linspace(.004,.00025,16):
 q=outline(rad)
 if q is not None and np.max(abs(np.max(q@ns.T,axis=0)-bounds))<1e-9:chosen=(rad,q);break
assert chosen is not None
rad,xy=chosen;center=(lo[:2]+hi[:2])/2
knots=np.array([0,.03,.08,.15,.30,.50,.70,.78,.85,.93,.98,1.]);scales=np.array([.72,.78,.82,.86,.90,.95,.99,1.,.98,.83,.50,0.]);curve=PchipInterpolator(knots,scales)
levels=np.unique(np.r_[np.linspace(0,1,20),knots]);verts=[]
for t in levels:
 scale=float(curve(t));shift=(1-scale)*np.array([.0003*np.sin(2.3*t),-.0002*np.sin(1.7*t)])
 pts=center+(xy-center)*scale+shift
 if t==1:pts=pts[:1]
 verts.extend(np.c_[pts,np.full(len(pts),lo[2]+t*(hi[2]-lo[2]))])
v=np.array(verts);h=ConvexHull(v);faces=h.simplices.copy()
for i,f in enumerate(faces):
 if np.dot(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]),h.equations[i,:3])<0:faces[i]=f[[0,2,1]]
used=np.unique(faces);mapping=np.full(len(v),-1);mapping[used]=np.arange(len(used));v=v[used];faces=mapping[faces]
assert np.max(abs(v.min(0)-lo))<1e-9 and np.max(abs(v.max(0)-hi))<1e-9
assert np.max(abs(np.max(v[:,:2]@ns.T,axis=0)-bounds))<1e-9
mesh.set('vertex',' '.join(f'{x:.12g}' for x in v.ravel()));mesh.set('face',' '.join(map(str,faces.ravel())))
new=A/'object_topwide_v1.xml';r.write(new,encoding='utf-8',xml_declaration=True)
m0=mj.MjModel.from_xml_path(str(original));m1=mj.MjModel.from_xml_path(str(new));i0=m0.body('item').id;i1=m1.body('item').id
assert abs(m0.body_mass[i0]-m1.body_mass[i1])<1e-12
# Verify every XML element except target mesh is byte-equivalent after normalization.
e0=E.parse(original);e1=E.parse(new)
for tree in [e0,e1]:tree.find("asset/mesh[@name='peb0']").attrib.clear()
assert E.tostring(e0.getroot())==E.tostring(e1.getroot())
report=dict(original=str(original),variant=str(new),changed='target peb0 mesh only; base foot, neighbors, tray, materials, free joints and initial poses unchanged',mass_g=float(m1.body_mass[i1]*1000),mesh_bounds_local_mm=[(lo*1000).tolist(),(hi*1000).tolist()],mesh_dimensions_mm=((hi-lo)*1000).tolist(),mesh_dimensions_initial_world_mm=(np.ptp(np.c_[v[:,:2]@rot.T,v[:,2]],axis=0)*1000).tolist(),all_8_xy_support_errors_mm=float(np.max(abs(np.max(v[:,:2]@ns.T,axis=0)-bounds))*1000),rounding_radius_mm=float(rad*1000),vertices=len(v),triangles=len(faces),profile_t=knots.tolist(),profile_relative_width=scales.tolist(),inertia_original=m0.body_inertia[i0].tolist(),inertia_variant=m1.body_inertia[i1].tolist(),inertia_note='recomputed from changed geometry at the same mass; not artificially held equal',original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),variant_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
(O/'topwide_asset_audit.json').write_text(json.dumps(report,indent=2));np.savez_compressed(O/'topwide_mesh_preview.npz',original=old,vertices=v,faces=faces);print(json.dumps(report,indent=2))
