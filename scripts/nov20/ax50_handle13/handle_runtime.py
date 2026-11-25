"""AX50 shared Newton world with the unchanged free vessel and passive bail."""
from pathlib import Path
import sys,json
ROOT=Path('/home/zjh/Evolution_PC');sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_leaf04'))
# Reuses the reviewed wrist PD and pair-specific palm damping; no leaf physics is invoked.
from leaf_runtime import damped_adapter,newton,np,wp
OUT=ROOT/'outputs/nov20/ax50_handle13'
class HandleRuntime(damped_adapter.AX50BrushRuntime):
 def _add_brush(self,b):
  b.default_joint_cfg.target_ke=0.;b.default_joint_cfg.target_kd=0.
  from newton._src.sim.builder import _BuilderShapeCollisionFilterPairs
  if isinstance(b._shape_collision_filter_pairs,list):
   old=list(b._shape_collision_filter_pairs);b._shape_collision_filter_pairs=_BuilderShapeCollisionFilterPairs()
   for pair in old:b._shape_collision_filter_pairs.append(pair)
  b.add_mjcf(str(OUT/'object.xml'),collapse_fixed_joints=False);self.prop_body_count=b.body_count-self.hand_body_count
  self.object_bodies=[i for i,n in enumerate(b.body_label) if str(n).rsplit('/',1)[-1] in ['item','bail']]
  self.prop_dof_start=self.hand_dof_count
  for d in range(self.prop_dof_start,b.joint_dof_count):
   b.joint_target_mode[d]=newton.JointTargetMode.NONE;b.joint_target_ke[d]=0.;b.joint_target_kd[d]=0.
  j=next(j for j,n in enumerate(b.joint_label) if str(n).endswith('/hinge'));self.hinge_q=int(b.joint_q_start[j]);self.hinge_dof=int(b.joint_qd_start[j])
 def _build(self,*args):
  super()._build(*args);self.vbd.extra_palm_shapes=[i for i,n in enumerate(self.model.shape_label) if str(n).endswith(('/link_2_0/palm_tile_12','/link_2_0/palm_tile_14'))]
if __name__=='__main__':
 h=HandleRuntime(side='left',wrist='six_axis',position=(.25,.25,.35),gravity=(0,0,-9.81));start=h.state.body_q.numpy()[h.object_bodies].copy();h.advance(.05,dt=.000125);m=h.model
 result=dict(scope='shared-world import and 0.05 s step only; hand is far away; not task acceptance',time=h.time,finite=all(np.isfinite(x.numpy()).all() for x in [h.state.joint_q,h.state.joint_qd,h.state.body_q,h.state.particle_q,h.state.particle_qd]),object_bodies=h.object_bodies,object_mass_kg=m.body_mass.numpy()[h.object_bodies].tolist(),object_start=start.tolist(),object_end=h.state.body_q.numpy()[h.object_bodies].tolist(),hinge_q_index=h.hinge_q,hinge_dof_index=h.hinge_dof,hinge_rad=float(h.state.joint_q.numpy()[h.hinge_q]),hinge_damping=float(m.joint_damping.numpy()[h.hinge_dof]),prop_position_gains=m.joint_target_ke.numpy()[h.prop_dof_start:].tolist(),total_dof=m.joint_dof_count,total_q=len(h.state.joint_q.numpy()),total_bodies=m.body_count)
 (OUT/'shared_smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
