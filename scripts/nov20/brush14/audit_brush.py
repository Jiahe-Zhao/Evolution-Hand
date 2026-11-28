
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,re,json
from pathlib import Path
import numpy as np,mujoco as mj
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1];D=np.load(P/(tag+'.npz'));m=mj.MjModel.from_xml_path(str(P/(tag+'.xml')));d=mj.MjData(m)
labels={};ids=[]
for i in range(m.ngeom):
 body=m.body(m.geom_bodyid[i]).name;match=re.fullmatch(r'link_([1-5])_([1-3])',body)
 if match:labels[i]=(int(match[1]),body);ids.append(i)
worst=(0,None);minclear=(1,None);hair_overlap=0.;sampled=0
for t in range(0,len(D['q']),16):
 d.qpos[:]=D['q'][t];mj.mj_forward(m,d);p=d.geom_xpos[ids];bounds=m.geom_rbound[ids];near=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)<bounds[:,None]+bounds[None,:]+.0001
 for a,b in zip(*np.where(np.triu(near,1))):
  i,j=ids[a],ids[b]
  if labels[i][0]==labels[j][0]:continue
  distance=mj.mj_geomDistance(m,d,i,j,.003,np.zeros(6))
  if distance<minclear[0]:minclear=(distance,[t,float(D['a'][t,0]),m.geom(i).name,m.geom(j).name,labels[i][1],labels[j][1]])
 for c in d.contact:
  i,j=c.geom
  if m.geom(i).name.startswith('bristle_') and m.geom(j).name.startswith('bristle_'):hair_overlap=max(hair_overlap,-float(c.dist))
 sampled+=1
out={'sample_rate_hz':30,'sampled_frames':sampled,'cross_finger_distal_min_distance_mm':minclear[0]*1000,'closest_pair':minclear[1],'cross_bundle_max_geometric_overlap_mm':hair_overlap*1000,'free_brush_joint':m.joint('brush_free').type.tolist(),'no_brush_motor':all(not m.joint(m.actuator_trnid[a,0]).name.startswith('bristle') and m.joint(m.actuator_trnid[a,0]).name!='brush_free' for a in range(m.nu)),'equality_constraint_count':m.neq}
(P/(tag+'_geometry_audit.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
