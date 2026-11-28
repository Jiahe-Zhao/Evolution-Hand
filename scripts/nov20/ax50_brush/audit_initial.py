
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
import sys,json,os,xml.etree.ElementTree as ET,numpy as np,time
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush'
sys.path.insert(0,str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
import writing_adapter as adapter
from ax50_visualskin import HandRuntime
import newton,warp as wp
class WritingRuntime(adapter.AX50BrushRuntime):
 def _add_brush(self,b):
  start=b.shape_count;super()._add_brush(b)
  for i in range(start,b.shape_count):b.shape_material_ke[i]=20000.;b.shape_material_kd[i]=10.
fit=json.loads((P/'grip_candidate_compact.json').read_text());cx,cy=fit['rod_xy'];root=_evo_parse_xml(ROOT/'outputs/nov20/brush14/verified.xml');world=root.getroot().find('worldbody');brush=world.find("body[@name='brush']");brush.set('pos',f'{cx} {cy} .125');brush.set('quat','1 0 0 0');brush.attrib.pop('euler',None)
world.find("geom[@name='paper']").set('pos',f'{cx+.025} {cy} .060');world.find("geom[@name='paper']").set('size','.09 .06 .001')
world.find("geom[@name='table']").set('pos',f'{cx+.025} {cy} .056');world.find("geom[@name='table']").set('size','.10 .07 .003')
xml=P/'tongji_scene_source.xml';root.write(xml)
r=WritingRuntime(side='left',wrist='six_axis',position=(0,0,.14),gravity=(0,0,-9.81),brush_xml=xml)
z=np.load(P/'prepared_hand_locked.npz');q=r.state.joint_q.numpy();oldnames=list(z['action_names'])+['tissue'];mapping=[r.joint_names.index(str(n)) for n in oldnames];q[mapping]=z['joint_q'];r.state.joint_q.assign(q);r.state.joint_qd.zero_();r.state.particle_q.assign(z['particle_q']);r.state.particle_qd.zero_();newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);r.state.body_qd.zero_()
for key,value in vars(r.state).items():
 dest=getattr(r.next,key,None)
 if isinstance(value,wp.array) and isinstance(dest,wp.array):wp.copy(dest,value)
r.rigid.reset(r.state,flags=0);r.vbd.reset(r.state,flags=0);r.collision.reset_contact_matching();r.feedback.zero_();r._advance_graph=None
targ=r.target.numpy();targ[mapping]=z['target'];r.target.assign(targ)

import mujoco as mj
r.rigid._update_mjc_data(r.rigid.mjw_data,r.model,r.state)
mm=r.rigid.mj_model;dd=mj.MjData(mm);dd.qpos[:]=r.rigid.mjw_data.qpos.numpy()[0];mj.mj_forward(mm,dd)
contacts=sorted([{'pair':[mm.geom(c.geom1).name,mm.geom(c.geom2).name],'depth_mm':float(-c.dist*1000)} for c in dd.contact if c.dist<0],key=lambda x:-x['depth_mm'])
(P/'initial_contacts.json').write_text(json.dumps(contacts,indent=2));print(contacts[:15],flush=True)
np.savez(P/'grasp_trial.npz',body_q=np.array([r.state.body_q.numpy()]),particle_q=np.array([r.state.particle_q.numpy()]),body_names=r.body_names_all)
(P/'grasp_trial.json').write_text(json.dumps({'status':'initial diagnostic geometry, NOT successful grasp','rows':[],'brush_info':r.brush_info(),'source_xml':str(xml),'hand_mount':[0,0,.14],'paper_top_z':.061,'brush_root_body':r.brush_root_body},indent=2))
