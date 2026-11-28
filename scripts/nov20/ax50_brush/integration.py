"""AX50 v3.0.0 + free brush, shared solver scene. Not a validated writing policy."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,hashlib,numpy as np
import sys
sys.path.insert(0,(str(_EVOLUTION_ROOT) + '/outputs/ax50_visualskin_v3_0_0'))
from adapter import AX50BrushRuntime
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_brush'))
class AX50WritingScene(AX50BrushRuntime):
 def _add_brush(self,b):
  first=b.shape_count;super()._add_brush(b)
  for i in range(first,b.shape_count):b.shape_material_ke[i]=20000.;b.shape_material_kd[i]=10.
  shaft=b.shape_label.index('brush_shaft');self.restored_connected_filters=[]
  for bundle in range(13):
   caps=[b.shape_label.index(f'bristle_{bundle:02d}_{seg:02d}') for seg in range(5)]
   for a,c in [(shaft,caps[0])]+list(zip(caps[:-1],caps[1:])):
    b.add_shape_collision_filter_pair(a,c);self.restored_connected_filters.append([b.shape_label[a],b.shape_label[c]])
 def restore_prepared_hand(self,z):
  import newton,warp as wp
  names=list(z['action_names'])+['tissue'];ids=[self.joint_names.index(str(n)) for n in names];q=self.state.joint_q.numpy();q[ids]=z['joint_q'];self.state.joint_q.assign(q);self.state.joint_qd.zero_();self.state.particle_q.assign(z['particle_q']);self.state.particle_qd.zero_();newton.eval_fk(self.model,self.state.joint_q,self.state.joint_qd,self.state);self.state.body_qd.zero_()
  self.rigid.reset(self.state,flags=0);self.vbd.reset(self.state,flags=0);self.collision.reset_contact_matching();self.feedback.zero_();self._advance_graph=None
  for key,value in vars(self.state).items():
   dst=getattr(self.next,key,None)
   if isinstance(value,wp.array) and isinstance(dst,wp.array):wp.copy(dst,value)
  target=self.target.numpy();target[ids]=z['target'];self.target.assign(target)
if __name__=='__main__':
 h=AX50WritingScene(side='left',wrist='six_axis',position=(0,0,.14),gravity=(0,0,0),brush_xml=P/'tongji_scene_source.xml');start=h.state.body_q.numpy().copy();h.advance(.025);v,f=h.visual_skin().update(h.state.body_q)
 out={'status':'import_and_short_step_passed','writing_status':'not passed; stable grasp unresolved','text':'同济','stroke_count':15,'seconds':h.time,'finite':bool(np.isfinite(h.state.body_q.numpy()).all() and np.isfinite(h.state.particle_q.numpy()).all() and np.isfinite(v).all()),'hand_action_count':len(h.action_names),'soft_particles':h.model.particle_count,'display_vertices':len(v),'brush':h.brush_info(),'restored_connected_filter_count':len(h.restored_connected_filters),'initialization':'open hand, zero gravity; not grasp or load validation','max_body_translation_m':float(np.linalg.norm(h.state.body_q.numpy()[:,:3]-start[:,:3],axis=1).max()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(P/'integration_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));assert out['finite'];print(json.dumps(out,ensure_ascii=False),flush=True)
