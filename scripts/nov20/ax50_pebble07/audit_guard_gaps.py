"""Offline convex distances at saved Newton body poses; not native contact depth."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,mujoco as mj,json,re,sys
from scipy.spatial.transform import Rotation
O=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));tag=sys.argv[1]
z=np.load(O/(tag+'.npz'),allow_pickle=True);m=mj.MjModel.from_binary_path(str(O/'penetration_runtime_model.mjb'));d=mj.MjData(m);mj.mj_forward(m,d)
norm=lambda s:re.sub('[^a-zA-Z0-9]','',str(s))
lookup={norm(s):i for i,s in enumerate(z['body_names'])};lookup.update({norm(str(s).split('/')[-1]):i for i,s in enumerate(z['body_names'])})
groups={k:[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name.endswith(s) and (m.geom_contype[g] or m.geom_conaffinity[g])] for k,s in [('thumb','link_1_2'),('index','link_2_3')]}
stone=[g for g in range(m.ngeom) if m.geom(g).name in ['stone0','foot0'] or any(x in m.geom(g).name for x in ['/stone0_','/foot0_'])]
assert stone and groups['thumb'] and groups['index']
selected=stone+groups['thumb']+groups['index'];rows=[]
for k,b in enumerate(z['body_q']):
 for g in selected:
  bn=m.body(int(m.geom_bodyid[g])).name;idx=lookup.get(norm(bn),lookup.get(norm(bn.split('/')[-1])))
  assert idx is not None,bn
  r=Rotation.from_quat(b[idx,3:]);d.geom_xpos[g]=b[idx,:3]+r.apply(m.geom_pos[g]);d.geom_xmat[g]=(r*Rotation.from_quat(m.geom_quat[g][[1,2,3,0]])).as_matrix().ravel()
 row=[float(z['time'][k])]
 for key in ['thumb','index']:row.append(min(mj.mj_geomDistance(m,d,g,s,.1,None)*1000 for g in groups[key] for s in stone))
 rows.append(row)
a=np.array(rows);np.savetxt(O/(tag+'_rigid_gaps.csv'),a,delimiter=',',header='time_s,thumb_gap_mm,index_gap_mm',comments='')
r=dict(scope=__doc__,samples=len(a),min_thumb_gap_mm=float(a[:,1].min()),min_index_gap_mm=float(a[:,2].min()),worst_thumb_time_s=float(a[a[:,1].argmin(),0]),final_thumb_gap_mm=float(a[-1,1]),final_index_gap_mm=float(a[-1,2]))
(O/(tag+'_rigid_gap_audit.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r))
