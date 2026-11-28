"""Recompute collision/converted constraints at recorded poses; no dynamics/control step."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from scipy.spatial.transform import Rotation
c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text())
h=PebbleRuntime(side='left',wrist='six_axis',position=np.asarray(c['position'])+[0,0,.05],orientation=c['orientation'],gravity=(0,0,-9.81),object_xml=Path(__file__).parent/'assets/object_topwide_v1.xml')
z=np.load(OUT/'shape_topwide_v1.npz');sb=h.model.shape_body.numpy();labels=list(map(str,h.model.shape_label));body=list(map(str,h.model.body_label));thumb=next(i for i,n in enumerate(body) if n.endswith('/link_1_2'))
geomshape=h.rigid.mjc_geom_to_newton_shape.numpy().reshape(-1)
shape_audit=[]
flags=h.model.shape_flags.numpy();scale=h.model.shape_scale.numpy();tr=h.model.shape_transform.numpy()
for i,label in enumerate(labels):
 if '/link_1_2/' not in label:continue
 src=h.model.shape_source[i]
 shape_audit.append(dict(shape=i,label=label,flags=int(flags[i]),scale=scale[i].tolist(),transform=tr[i].tolist(),vertex_count=len(src.vertices) if src is not None else 0))
h.rigid._mujoco.mj_saveModel(h.rigid.mj_model,str(OUT/'penetration_runtime_model.mjb'),None)
(OUT/'penetration_shape_flags.json').write_text(json.dumps(dict(shapes=shape_audit,flag_values={k:int(v) for k,v in newton.ShapeFlags.__members__.items()}),indent=2))
from collections import Counter
from scipy.spatial import ConvexHull
topo=[]
for shape in [74,next(i for i,n in enumerate(labels) if n.endswith('/item/stone0'))]:
 mesh=h.model.shape_source[shape];v=np.asarray(mesh.vertices);f=np.asarray(mesh.indices).reshape(-1,3)
 edges=Counter(tuple(sorted((int(a),int(b)))) for face in f for a,b in zip(face,np.roll(face,-1)))
 signed=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
 normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
 outward=np.einsum('ij,ij->i',normals,v[f].mean(1)-v.mean(0))
 topo.append(dict(label=labels[shape],vertices=len(v),faces=len(f),boundary_edges=sum(n==1 for n in edges.values()),nonmanifold_edges=sum(n>2 for n in edges.values()),signed_volume_mm3=signed*1e9,convex_hull_volume_mm3=float(ConvexHull(v).volume*1e9),inward_centroid_faces=int(sum(outward< -1e-14)),outward_centroid_faces=int(sum(outward>1e-14))))
 np.savez_compressed(OUT/f'penetration_mesh_{shape}.npz',vertices=v,faces=f)
(OUT/'penetration_mesh_topology.json').write_text(json.dumps(topo,indent=2))
rows=[]
for frame in [190,220,225,230,235,240,300,475]:
 h.state.body_q.assign(z['body_q'][frame]);h.state.joint_q.assign(z['joint_q'][frame]);p=h.state.particle_q.numpy();p[:h.real_count]=z['particles'][frame];h.state.particle_q.assign(p)
 h.collision.collide(h.state,h.raw_contacts);raw=h.raw_contacts
 n=int(raw.rigid_contact_count.numpy()[0]);a=raw.rigid_contact_shape0.numpy()[:n];b=raw.rigid_contact_shape1.numpy()[:n];nn=raw.rigid_contact_normal.numpy()[:n];p0=raw.rigid_contact_point0.numpy()[:n];p1=raw.rigid_contact_point1.numpy()[:n];bq=h.state.body_q.numpy()
 pairs=[]
 for k in range(n):
  i,j=int(a[k]),int(b[k]);bi,bj=int(sb[i]),int(sb[j])
  if set([bi,bj])!={thumb,h.item}:continue
  def world(b,p):return p if b<0 else Rotation.from_quat(bq[b,3:]).apply(p)+bq[b,:3]
  w0,w1=world(bi,p0[k]),world(bj,p1[k]);normal=nn[k]
  pairs.append(dict(shape0=labels[i],shape1=labels[j],normal=normal.tolist(),normal_length=float(np.linalg.norm(normal)),point0_world_mm=(w0*1000).tolist(),point1_world_mm=(w1*1000).tolist(),signed_point_separation_mm=float((w1-w0)@normal*1000)))
 h.rigid._convert_contacts_to_mjwarp(h.model,h.state,raw)
 d=h.rigid.mjw_data;nc=int(d.nacon.numpy().reshape(-1)[0]);g=d.contact.geom.numpy()[:nc];dist=d.contact.dist.numpy()[:nc];nf=d.contact.frame.numpy()[:nc];converted=[]
 for k in range(nc):
  i,j=[int(geomshape[int(gi)]) for gi in g[k]]
  if min(i,j)<0:continue
  if set([int(sb[i]),int(sb[j])])!={thumb,h.item}:continue
  converted.append(dict(shape0=labels[i],shape1=labels[j],dist_mm=float(dist[k]*1000),frame=np.asarray(nf[k]).tolist()))
 r=dict(time_s=float(z['time'][frame]),frame=frame,total_raw_contacts=n,total_converted_contacts=nc,raw_thumb_target=pairs,converted_thumb_target=converted)
 rows.append(r)
 print('FRAME',frame,'raw',len(pairs),'converted',len(converted),'depth_range_mm',([min(x['dist_mm'] for x in converted),max(x['dist_mm'] for x in converted)] if converted else None),flush=True)
(OUT/'penetration_static_contacts.json').write_text(json.dumps(dict(scope=__doc__,frames=rows),indent=2))
print('STATIC_CONTACT_PROBE_COMPLETE',flush=True)
