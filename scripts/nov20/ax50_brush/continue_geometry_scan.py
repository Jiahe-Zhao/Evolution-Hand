
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path((str(_EVOLUTION_ROOT)))
P=ROOT/'outputs/nov20/ax50_brush'
OUT=P/'continue'
sys.path.insert(0,str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
import ax50_visualskin
sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
from continue_opposed_trial import prepare,Q,MOUNT,R
import newton

r=AX50WritingScene(side='left',wrist='locked',position=MOUNT,orientation=Q,gravity=(0,0,0),iterations=60,brush_xml=P/'tongji_scene_source.xml')
r.restore_prepared_hand(np.load(prepare()))
body=r.state.body_q.numpy()
shape_body=r.model.shape_body.numpy()
shape_transform=r.model.shape_transform.numpy()
bone_names=np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']]
particle=r.state.particle_q.numpy()

def transform(p,points):
    return Rotation.from_quat(p[3:]).apply(points)+p[:3]

def world_mesh(shape):
    mesh=r.model.shape_source[shape]
    return transform(body[shape_body[shape]],transform(shape_transform[shape],np.asarray(mesh.vertices,dtype=float)))

def local(points):
    return R.inv().apply(points-MOUNT)

def radial(points,center):
    delta=points-center
    return np.linalg.norm(delta[:,:2],axis=1)-.005

def gap(points,center):
    rr=radial(points,center)
    p=points[np.argmin(rr)]
    return float(rr.min()),local(p).tolist()

meshes={}
for shape,label in enumerate(r.model.shape_label):
    if not label.startswith('/ax50_left/') or not isinstance(r.model.shape_source[shape],newton.Mesh):
        continue
    short=label.rsplit('/',2)[-2]+'/'+label.rsplit('/',1)[-1]
    meshes[label]=short
points={label:local(world_mesh(shape)) for shape,label in enumerate(r.model.shape_label) if label in meshes}
for bone in ['link_2_1','link_2_2','link_2_3']:
    points[bone]=local(particle[np.flatnonzero(bone_names==bone)+r.pad_offset])
print('EXTENTS',json.dumps({key:[p.min(0).tolist(),p.max(0).tolist()] for key,p in points.items() if key.endswith(('skin_1_2','skin_3_3')) or key.startswith('link_2_')}),flush=True)
keep=[key for key in points if key.endswith(('skin_1_2','skin_3_3'))]
reject=[key for key in points if '/link_1_0/' in key or '/link_2_0/' in key or '/link_0_' in key or key.endswith('palm_chassis')]
rows=[]
for x in np.arange(.020,.043,.0005):
 for y in np.arange(.030,.071,.0005):
  center=np.array([x,y,0.])
  good={key:gap(points[key],center) for key in keep}
  soft=min(gap(points[key],center)[0] for key in ['link_2_1','link_2_2','link_2_3'])
  other=min((gap(points[key],center)[0],key) for key in reject)
  if -.0015<good[keep[0]][0]<.0003 and -.0008<soft<.0004 and other[0]>.0003:
   rows.append({'x':x,'y':y,'thumb':good[keep[0]],'middle':good[keep[1]],'soft_gap_mm':soft*1000,'closest_other':other})
rows.sort(key=lambda d:abs(d['thumb'][0]+d['soft_gap_mm']/1000)+max(0,abs(d['middle'][0]+.0005)-.002)*.3)
(OUT/'geometry_scan.json').write_text(json.dumps({'pen_axis_local':[0,0,-1],'radius_m':.005,'large_contact_y_span_m':.13,'candidates':rows},indent=2))
print('CANDIDATES',len(rows),flush=True)
for row in rows[:30]:print(json.dumps(row),flush=True)
