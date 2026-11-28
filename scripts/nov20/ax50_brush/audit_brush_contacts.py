
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json
from pathlib import Path
import numpy as np
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_brush'));sys.path.insert(0,(str(_EVOLUTION_ROOT) + '/scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
r=AX50WritingScene(side='left',wrist='locked',position=(0,0,.14),gravity=(0,0,-9.81),brush_xml=P/'tongji_scene_source.xml');r.restore_prepared_hand(np.load(P/'prepared_hand_locked.npz'));r.advance(.001)
c=r.rigid_observation;n=int(c.rigid_contact_count.numpy()[0]);sb=r.model.shape_body.numpy();labels=r.model.shape_label;out=[]
for i,(a,b,f) in enumerate(zip(c.rigid_contact_shape0.numpy()[:n],c.rigid_contact_shape1.numpy()[:n],c.force.numpy()[:n])):
 if a<0 or b<0:continue
 if 'brush_shaft' not in (labels[a],labels[b]):continue
 out.append(dict(shapes=[labels[a],labels[b]],force=f.tolist()))
c=r.contacts;n=int(c.soft_contact_count.numpy()[0]);s=c.soft_contact_shape.numpy()[:n];f=r.out_f.numpy()[:n];sel=np.array([labels[i]=='brush_shaft' for i in s]);print('RIGID',json.dumps(out));print('SOFT',json.dumps({'count':int(sel.sum()),'force_on_soft_N':f[sel].sum(0).tolist(),'sum_magnitudes_N':float(np.linalg.norm(f[sel],axis=1).sum()),'max_depth_mm':float(r.out_depth.numpy()[:n][sel].max()*1000) if sel.any() else 0}));print('ROOT',r.state.body_q.numpy()[r.brush_root_body].tolist())
