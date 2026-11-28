
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
def trial(x,y,name,seconds=.3,dt=.00025,overrides=None,gravity=-9.81):
 place(x,y);r.model.gravity.assign(np.array([[0,0,gravity]],dtype=np.float32));r.rigid.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES);start=r.state.body_q.numpy()[r.brush_root_body].copy();target=baset.copy()
 for key,val in (overrides or {}).items():target[r.joint_names.index(key)]+=val
 rows=[];frames=[r.state.body_q.numpy()];particles=[r.state.particle_q.numpy()];wall=time.time()
 for k in range(round(seconds/.005)):
  r.target.assign(baset+(target-baset)*min(1,(k+1)/10));r.advance(.005,dt=dt);b=r.state.body_q.numpy();p=r.state.particle_q.numpy();finite=bool(np.isfinite(b).all() and np.isfinite(p).all());disp=float(np.linalg.norm(b[r.brush_root_body,:3]-start[:3])*1000);soft=r.tactile();rigid=r.rigid_tactile();sn=float(soft['external_window_peak_sum_force_magnitudes_N'].max());rn=float(rigid['external_window_peak_sum_force_magnitudes_N'].max());row={'t':r.time,'disp_mm':disp,'soft_peak_N':sn,'rigid_peak_N':rn,'finite':finite};rows.append(row)
  if finite:frames.append(b);particles.append(p)
  if k%10==0 or not finite or disp>20:print(name,row,'wall',round(time.time()-wall,2),flush=True)
  if not finite or disp>20:break
 np.savez(O/(name+'.npz'),body_q=frames,particle_q=particles,body_names=r.body_names_all,state_joint_q=r.state.joint_q.numpy(),state_joint_qd=r.state.joint_qd.numpy(),state_particle_q=r.state.particle_q.numpy(),state_particle_qd=r.state.particle_qd.numpy(),state_body_q=r.state.body_q.numpy(),state_body_qd=r.state.body_qd.numpy(),target=r.target.numpy())
 (O/(name+'.json')).write_text(json.dumps({'name':name,'x':x,'y':y,'gravity':gravity,'dt':dt,'overrides':overrides,'rows':rows},indent=2));print('DONE',name,rows[-1],flush=True);return rows
