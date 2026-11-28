
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from leaf_runtime import *
import json,csv
b=newton.ModelBuilder(gravity=(0,0,-9.81));newton.solvers.SolverMuJoCo.register_custom_attributes(b);b.default_joint_cfg.target_ke=0;b.default_joint_cfg.target_kd=0;b.add_mjcf(str(CLEAN_XML),collapse_fixed_joints=False)
for mesh in b.shape_source:
 if isinstance(mesh,newton.Mesh):mesh.build_sdf(device='cuda:0',target_voxel_size=.00025,narrow_band_range=(-.002,.002),margin=.002)
for j in range(len(b.joint_target_ke)):b.joint_target_ke[j]=b.joint_target_kd[j]=0;b.joint_target_mode[j]=newton.JointTargetMode.NONE
m=b.finalize(device='cuda:0');ids=[j for j,n in enumerate(m.joint_label) if str(n).rsplit('/',1)[-1].startswith('hinge')];qi=[int(m.joint_q_start.numpy()[j]) for j in ids];di=[int(m.joint_qd_start.numpy()[j]) for j in ids]
results=[];dt=1/480;smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
for k in range(2):
 s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);control=m.control();solver=newton.solvers.SolverMuJoCo(m,use_mujoco_contacts=False,integrator='implicitfast',nconmax=512,njmax=8192);pipe=newton.CollisionPipeline(m,broad_phase='sap',rigid_contact_max=512,soft_contact_max=1);c=pipe.contacts();rows=[]
 for i in range(1056):
  t=i*dt;q=s.joint_q.numpy();f=np.zeros(m.joint_dof_count,np.float32);f[di]=-.0014*q[qi]-.04*(q[qi]-np.clip(q[qi],0,1.58));f[di[k]]+=.0038*smooth((t-.15)/.4)*(1-smooth((t-1.1)/.2));control.joint_f.assign(f);s.clear_forces();pipe.collide(s,c);solver.step(s,n,control,c,dt);s,n=n,s;rows.append([t+dt,*s.joint_q.numpy()[qi]])
 a=np.array(rows);np.savetxt(OUT_DIR/f'object_probe_{k}.csv',a,delimiter=',',header='time_s,leaf0_rad,leaf1_rad',comments='');r=dict(leaf=k,peak_deg=float(np.rad2deg(a[:,k+1]).max()),other_peak_abs_deg=float(np.abs(np.rad2deg(a[:,2-k])).max()),end_deg=np.rad2deg(a[-1,1:]).tolist(),finite=bool(np.isfinite(a).all()));results.append(r);print(r,flush=True)
(OUT_DIR/'object_probe.json').write_text(json.dumps(dict(scope='object-only Newton MuJoCo-Warp; no hand acceptance',results=results),indent=2))
