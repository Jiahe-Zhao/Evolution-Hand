
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush';sys.path.insert(0,str(ROOT/'outputs/ax50_visualskin_v3_0_0'))
from ax50_visualskin import HandRuntime
import warp as wp
h=HandRuntime(side='left',wrist='locked',position=(0,0,.18),gravity=(0,0,0));candidate=json.loads((P/'grip_candidate_opposed.json').read_text());target=dict(zip(candidate['joint_names'],candidate['native_q']));initial=h.observe()['joint_position'];desired=np.array([target[n] for n in h.action_names]);frames=[];t0=time.time()
for k in range(80):
 u=min(1,(k+1)/55);u=u*u*(3-2*u);h.set_targets(initial+(desired-initial)*u);h.advance(.01)
 if k%4==0:frames.append(h.state.body_q.numpy());print(k,h.time,time.time()-t0,flush=True)
 if not np.isfinite(h.state.particle_q.numpy()).all():raise RuntimeError('nonfinite tissue')
np.savez(P/'prepared_hand_opposed.npz',joint_q=h.state.joint_q.numpy(),joint_qd=h.state.joint_qd.numpy(),particle_q=h.state.particle_q.numpy(),particle_qd=h.state.particle_qd.numpy(),body_q=h.state.body_q.numpy(),body_qd=h.state.body_qd.numpy(),frames=np.array(frames),body_names=h.body_names_all,action_names=h.action_names,target=h.target.numpy())
obs=h.observe();err=np.max(np.abs(obs['joint_position']-desired));(P/'prepared_hand_opposed.json').write_text(json.dumps({'time_s':h.time,'joint_error_max_rad':float(err),'candidate':candidate,'status':'prepared empty-hand posture; not brush grasp'},indent=2));print('DONE',err,flush=True)
