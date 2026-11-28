
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from env import episode
from build import OUT
import json,sys,numpy as np
import mujoco as mj
from PIL import Image
for t in ('pebble','scrape'):
 gen=episode(t,gpu=True);meta=next(gen);traj=[]
 for i in range(720 if t=='pebble' else 240):
    z=gen.send(meta['base']);traj.append(z['task_metrics'])
 if t=='pebble':assert traj[-1]['neighbor_max_mm']<5 and abs(traj[-1]['lift_mm'])<3,traj[-1]
 assert not any(k['success'] for k in traj)
 (OUT/(t+'_idle.json')).write_text(json.dumps(traj))
 m=mj.MjModel.from_xml_path(str(OUT/(t+'.xml')));d=mj.MjData(m);mj.mj_forward(m,d)
 for k in range(m.ngeom):
    if m.geom(k).name.startswith('handgeom'):m.geom_group[k]=5
 opt=mj.MjvOption();opt.geomgroup[5]=0
 cam=mj.MjvCamera();cam.lookat[:]=[-.096,0,.030];cam.azimuth=106;cam.elevation=-50;cam.distance=.205
 with mj.Renderer(m,height=400,width=560) as renderer:
    renderer.update_scene(d,cam,scene_option=opt);Image.fromarray(renderer.render()).save(OUT/(t+'_objects.png'))
 print(t,traj[-1],flush=True)
