from keys_runtime import *
import json
out=ROOT/'outputs/nov20/ax50_keys17'
h=KeysRuntime(side='right',wrist='locked',position=(0,0,.3),gravity=(0,0,-9.81))
print('ACTIONS',h.action_names,flush=True)
print('BODIES',h.body_names,flush=True)
print('LIMITS',list(zip(h.action_names,h.action_lower.tolist(),h.action_upper.tolist())),flush=True)
np.savez(out/'initial.npz',body_q=h.state.body_q.numpy(),body_names=h.body_names_all,particle_q=h.state.particle_q.numpy(),joint_q=h.state.joint_q.numpy())
import mujoco
mujoco.mj_saveModel(h.rigid.mj_model,str(out/'keys_ik.mjb'))
for i in range(20):
 h.advance(.01)
 print('SETTLE',h.time,h.state.joint_q.numpy()[h.key_q],flush=True)
print('DONE',flush=True)
