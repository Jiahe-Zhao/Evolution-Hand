
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os
os.environ.setdefault('MUJOCO_GL','egl')
from build import *
import sys,json,warp as wp,newton
from scipy.spatial.transform import Rotation as R
task=sys.argv[1];negative='negative' in sys.argv;tag=task+('_negative' if negative else '')
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003
b.add_mjcf(str(scene(task)),collapse_fixed_joints=False)
joints=[str(x).split('/')[-1] for x in b.joint_label]
for i,name in enumerate(joints):
 k=b.joint_qd_start[i]
 if 'diagnostic_wrist' in name:b.joint_target_ke[k]=1700;b.joint_target_kd[k]=42
 if name=='hinge':b.joint_target_ke[k]=0;b.joint_target_kd[k]=0
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);c=m.control();solver=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=40)
names=[str(a).split('/')[-1] for a in m.body_label]; palm=names.index('link_0_0');item=names.index('item');tips=[names.index('link_%d_3'%i) for i in range(1,6)]
base=c.joint_target_q.numpy().copy();trajectory=[];actions=[];frames=[];stats=[];dt=1/240
mm=solver.mj_model;dd=solver.mj_data
for i in range(mm.ngeom):
 g=mm.geom(i)
 if i%7==0 or i>mm.ngeom-4: print(i,g.name,mm.body(mm.geom_bodyid[i]).name,mm.geom_contype[i],mm.geom_conaffinity[i],dd.geom_xpos[i],mm.geom_size[i])
print(joints); print(mm.nu)
