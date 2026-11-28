
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
start=r.state.body_q.numpy()[r.brush_root_body].copy();initial=r.target.numpy().copy();goal=baset.copy()
for n,v in preload2.items():goal[r.joint_names.index(n)]+=3*v
rows=[];frames=[r.state.body_q.numpy()];particles=[r.state.particle_q.numpy()];wall=time.time();r.time=0
for k in range(520):
 t=(k+1)*.005;u=min(1,t/.3);r.target.assign(initial+(goal-initial)*(u*u*(3-2*u)));g=-9.81*np.clip((t-.3)/.3,0,1);r.model.gravity.assign(np.array([[0,0,g]],dtype=np.float32));r.rigid.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES);r.advance(.005);b=r.state.body_q.numpy();p=r.state.particle_q.numpy();fin=bool(np.isfinite(b).all() and np.isfinite(p).all());q=b[r.brush_root_body];disp=float(np.linalg.norm(q[:3]-start[:3])*1000);axis=Rot.from_quat(q[3:]).apply([0,0,1]);tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))));soft=r.tactile();rigid=r.rigid_tactile();force=soft['force_on_tissue_N'].sum(0)+rigid['force_on_body_N'].sum(0);row={'t':t,'gravity':float(g),'disp_mm':disp,'tilt_deg':tilt,'soft_peak_N':float(soft['external_window_peak_sum_force_magnitudes_N'].max()),'rigid_peak_N':float(rigid['external_window_peak_sum_force_magnitudes_N'].max()),'force_on_hand_N':force.tolist(),'finite':fin};rows.append(row)
 if k%2==0 and fin:frames.append(b);particles.append(p)
 if k%20==0 or not fin or disp>10:print('LOAD',row,'wall',round(time.time()-wall,1),flush=True)
 if not fin or disp>10:break
np.savez(O/'audit22_load_hold.npz',body_q=frames,particle_q=particles,body_names=r.body_names_all,state_joint_q=r.state.joint_q.numpy(),state_joint_qd=r.state.joint_qd.numpy(),state_particle_q=r.state.particle_q.numpy(),state_particle_qd=r.state.particle_qd.numpy(),state_body_q=r.state.body_q.numpy(),state_body_qd=r.state.body_qd.numpy(),target=r.target.numpy())
(O/'audit22_load_hold.json').write_text(json.dumps({'status':'measured load test; assess before acceptance','rows':rows,'start':start.tolist(),'contact':'vertex soft contact; force-space shaft/thumb'},indent=2));print('LOAD_DONE',rows[-1],flush=True)
