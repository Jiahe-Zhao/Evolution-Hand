
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from strike_runtime import *
import csv
h=construct();dt=.000125;period=.005
q0=np.array(h.state.joint_q.numpy());p0=h.state.particle_q.numpy().copy()
rows=[];poses=[];ps=[];times=[];targets=[];stop="completed"
fingers=np.array([[0,.7,.8,.6]]*4);thumb=np.array([.15,.2,.15,.5,.4])
for k in range(160):
 t=k*period;u=np.clip((t-.1)/.5,0,1);u=u*u*(3-2*u)
 target=hand_targets(h.action_names,fingers*u,thumb*u)
 h.set_targets(target);h.advance(period,dt=dt)
 b=h.state.body_q.numpy();p=h.state.particle_q.numpy();v=h.state.particle_qd.numpy();finite=np.isfinite(b).all() and np.isfinite(p).all()
 row=dict(time_s=h.time,closure=u,particle_speed_max_m_s=float(np.linalg.norm(v[:h.real_count],axis=1).max()),particle_displacement_max_mm=float(np.linalg.norm(p[:h.real_count]-p0[:h.real_count],axis=1).max()*1000),finite=bool(finite))
 rows.append(row);poses.append(b);ps.append(p[:h.real_count]);times.append(h.time);targets.append(target)
 if k%20==0:print(row,flush=True)
 if not finite or row["particle_speed_max_m_s"]>5:stop="soft_motion_guard_stop";break
np.savez_compressed(OUT/"continuous_hand.npz",time=times,body_q=poses,body_names=h.body_names_all,particles=ps,tri=h.model.tri_indices.numpy(),targets=targets)
with (OUT/"continuous_hand_trace.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
r=dict(stop=stop,time_s=h.time,scope="open initial state continuous closure; hammer far away, not grasp acceptance",max_particle_speed_m_s=max(x["particle_speed_max_m_s"] for x in rows),last=rows[-1]);(OUT/"continuous_hand_result.json").write_text(json.dumps(r,indent=2));print(r,flush=True)
