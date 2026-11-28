
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
  shaft=b.shape_label.index('brush_shaft')
  for bundle in range(13):
   caps=[b.shape_label.index(f'bristle_{bundle:02d}_{seg:02d}') for seg in range(5)]
   b.add_shape_collision_filter_pair(shaft,caps[0])
   for a,c in zip(caps[:-1],caps[1:]):b.add_shape_collision_filter_pair(a,c)

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
targ=r.target.numpy();targ[mapping]=z['target'];
for n,delta in [("digit_1_flex_1",-.08),("digit_2_flex_1",.015),("digit_3_flex_1",.015)]:targ[r.joint_names.index(n)]+=delta
r.target.assign(targ)
frames=[r.state.body_q.numpy()];particles=[r.state.particle_q.numpy()];rows=[];start=r.state.body_q.numpy()[r.brush_root_body,:3].copy();wall=time.time();steps=int(os.getenv('FRAMES','30'))
for k in range(steps):
 r.advance(.01);poses=r.state.body_q.numpy();soft=r.tactile();rigid=r.rigid_tactile();pos=poses[r.brush_root_body,:3];row={'t':r.time,'brush_xyz':pos.tolist(),'brush_displacement_mm':float(np.linalg.norm(pos-start)*1000),'hand_max_rigid_contact_N':float(np.max(rigid['external_window_peak_sum_force_magnitudes_N'])),'hand_max_soft_contact_N':float(np.max(soft['external_window_peak_sum_force_magnitudes_N'])),'wrist_q':r.state.joint_q.numpy()[:6].tolist()};rows.append(row);frames.append(poses);particles.append(r.state.particle_q.numpy());print(k,row['brush_displacement_mm'],row['hand_max_rigid_contact_N'],time.time()-wall,flush=True)
 if not np.isfinite(poses).all() or not np.isfinite(particles[-1]).all():
  frames.pop();particles.pop();print('NONFINITE STOP',flush=True);break
 if row['brush_displacement_mm']>60:break
np.savez(P/'grasp_trial_preload.npz',body_q=np.array(frames),particle_q=np.array(particles),body_names=r.body_names_all)
(P/'grasp_trial_preload.json').write_text(json.dumps({'status':'diagnostic grasp attempt, not accepted writing','rows':rows,'brush_info':r.brush_info(),'source_xml':str(xml),'hand_mount':[0,0,.14],'paper_top_z':.061,'brush_root_body':r.brush_root_body},indent=2))
print('DONE',flush=True)
