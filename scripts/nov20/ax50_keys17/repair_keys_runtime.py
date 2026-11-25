"""AX50 v3 physical hand plus unchanged scene17 passive keys."""
import sys
from pathlib import Path
ROOT=Path('/home/zjh/Evolution_PC')
sys.path.insert(0,str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_brush'))
from damped_adapter import AX50BrushRuntime
import damped_adapter
from wrist_runtime import wrist_pd
damped_adapter.control_pd_subset=wrist_pd
import newton, warp as wp
import numpy as np
CENTERS=np.array([[-.123,.001,.096],[-.128,-.018,.096],[-.118,-.037,.096],[-.097,-.056,.096]])
class KeysRuntime(AX50BrushRuntime):
 def _add_brush(self,b):
  cfg=newton.ModelBuilder.ShapeConfig(density=0,mu=.6,ke=20000,kd=10,has_particle_collision=True)
  self.key_bodies=[];self.key_q=[];self.key_dof=[]
  for k,pos in enumerate([[-.114,-.028,.080],*CENTERS]):
   half=(.035,.043,.002) if k==0 else (.009,.0075,.002)
   mass=0 if k==0 else .004
   inertia=np.diag([mass/3*(half[1]**2+half[2]**2),mass/3*(half[0]**2+half[2]**2),mass/3*(half[0]**2+half[1]**2)])
   x=wp.transform(tuple(pos),wp.quat_identity())
   body=b.add_link(xform=x,mass=mass,inertia=wp.mat33(inertia),label='deck' if k==0 else f'key{k}')
   b.add_shape_box(body,hx=half[0],hy=half[1],hz=half[2],cfg=cfg,label='deck' if k==0 else f'key{k}')
   if k==0:joint=b.add_joint_fixed(-1,body,parent_xform=x,label='deck_fixed')
   else:
    joint=b.add_joint_prismatic(-1,body,axis=(0,0,-1),parent_xform=x,limit_lower=0,limit_upper=.004,target_ke=0,target_kd=0,damping=.8,armature=.001,custom_attributes={'mujoco:dof_passive_stiffness':210.},label=f'key{k}_slide')
    self.key_bodies.append(body);self.key_q.append(b.joint_q_start[joint]);self.key_dof.append(b.joint_qd_start[joint])
   b.add_articulation([joint],label=f'key_assembly{k}')
  self.prop_body_count=5
