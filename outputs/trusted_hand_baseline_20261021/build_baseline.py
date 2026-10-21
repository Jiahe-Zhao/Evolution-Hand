"""Authorized pad-envelope revision in an isolated hand copy."""
import json,hashlib,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
work=Path('/home/zjh/Evolution_PC/outputs/trusted_hand_baseline_20261021')
source=work/'frozen/right';dest=work/'candidate/right'
if dest.exists():raise FileExistsError(dest)
shutil.copytree(source,dest);path=dest/'urdf/current_agent.urdf';tree=ET.parse(path);r=tree.getroot();before_joints=[ET.tostring(j) for j in r.findall('joint')];changes=[]
for link in r.findall('link'):
 collisions={c.get('name'):c for c in link.findall('collision')}
 if 'finger_pad' not in collisions:continue
 meshes={}
 for name,c in collisions.items():
  o=c.find('origin')
  if o is not None:assert o.get('xyz','0 0 0')=='0 0 0' and o.get('rpy','0 0 0')=='0 0 0'
  node=c.find('geometry/mesh');mesh=trimesh.load(path.parent/node.get('filename'),force='mesh');meshes[name]=mesh
 pad=meshes['finger_pad'];old_bounds=pad.bounds.copy();bearing=np.concatenate([m.bounds for name,m in meshes.items() if 'bearing' in name])
 lo=bearing.min(0)[:2]-.0005;hi=bearing.max(0)[:2]+.0005
 width=hi-lo;center=(hi+lo)/2;vertices=pad.vertices.copy()
 vertices[:,:2]=(vertices[:,:2]-old_bounds.mean(0)[:2])*(width/(old_bounds[1,:2]-old_bounds[0,:2]))+center
 pad.vertices=vertices;pad.export(path.parent/collisions['finger_pad'].find('geometry/mesh').get('filename'))
 old_inert=link.find('inertial');old_mass=float(old_inert.find('mass').get('value'))
 masses=[];centers=[];inertias=[]
 for name,mesh in meshes.items():
  mesh.density=2700. if 'bearing' in name else 1100.;masses.append(mesh.mass);centers.append(mesh.center_mass);inertias.append(mesh.moment_inertia)
 mass=sum(masses);com=sum(m*c for m,c in zip(masses,centers))/mass;tensor=np.zeros((3,3))
 for m,c,I in zip(masses,centers,inertias):
  d=c-com;tensor+=I+m*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
 eig=np.linalg.eigvalsh(tensor);assert eig.min()>0 and eig[-1]<=sum(eig[:2])+1e-12
 old_inert.find('mass').set('value',str(mass));old_inert.find('origin').set('xyz',' '.join(map(str,com)))
 old_inert.find('inertia').attrib.update({k:str(tensor[i,j]) for k,i,j in [('ixx',0,0),('ixy',0,1),('ixz',0,2),('iyy',1,1),('iyz',1,2),('izz',2,2)]})
 changes.append(dict(link=link.get('name'),old_bounds_m=old_bounds.tolist(),new_bounds_m=pad.bounds.tolist(),new_width_xy_mm=(width*1000).tolist(),old_mass_kg=old_mass,new_mass_kg=mass,principal_inertia_kg_m2=eig.tolist(),lateral_clearance_m=.0005))
assert before_joints==[ET.tostring(j) for j in r.findall('joint')]
tree.write(path,encoding='utf-8',xml_declaration=True)
manifest=json.loads((work/'baseline_manifest.json').read_text())
for name,sha in manifest['files_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
report=dict(authorization='User approved independent baseline pad widening, 2026-10-21',source_sha256=manifest['source_sha256'],candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changes=changes,joints_unchanged=True,source_unchanged=True,total_mass_kg=sum(float(l.find('inertial/mass').get('value')) for l in r.findall('link') if l.find('inertial/mass') is not None),status='geometry generated; awaiting runtime/contact tests',motor_mass_calibrated=False)
(work/'candidate_manifest.json').write_text(json.dumps(report,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
fig,axs=plt.subplots(1,5,figsize=(14,4),facecolor='#dce3eb')
for ax,item in zip(axs,changes):
 b=np.array(item['old_bounds_m'])*1000;n=np.array(item['new_bounds_m'])*1000;rad=b[0,2];span=b[1,2]+rad
 ax.set_facecolor('#dce3eb')
 for z in [0,span]:ax.add_patch(Circle((0,z),rad,color='#c68233',label='Bearing' if z==0 else None))
 ax.add_patch(Rectangle((n[0,1],n[0,2]),n[1,1]-n[0,1],n[1,2]-n[0,2],color='#369a86',alpha=.7,label='New pad'))
 ax.add_patch(Rectangle((b[0,1],b[0,2]),b[1,1]-b[0,1],b[1,2]-b[0,2],fill=False,edgecolor='#21324c',linestyle='--',linewidth=2,label='Old pad'))
 ax.set(xlim=(-5,5),ylim=(-4,21),title=item['link'],xlabel='Local Y (mm)');ax.set_aspect('equal');ax.grid(alpha=.2)
axs[0].set_ylabel('Local Z (mm)');axs[-1].legend(loc='upper right',fontsize=7)
fig.suptitle('Authorized baseline: pad extends 0.5 mm beyond bearing lateral envelope');fig.tight_layout();fig.savefig(work/'pad_revision.png',dpi=160)
print(json.dumps(report,indent=2))
