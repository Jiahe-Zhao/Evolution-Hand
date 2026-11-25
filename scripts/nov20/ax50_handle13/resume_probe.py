from contact_audit_handle import *
import sys
from scipy.spatial.transform import Rotation as R
tag=sys.argv[1];p=json.loads((OUT/(tag+'_model.json')).read_text());h=AuditedHandleRuntime(side='left',wrist='six_axis',position=p['mount'],orientation=p['plan']['orientation'],gravity=(0,0,-9.81));z=np.load(OUT/(tag+'_checkpoint.npz'));seq=np.load(OUT/(tag+'.npz'))
for state in [h.state,h.next]:
 for k in ['joint_q','joint_qd','body_q','body_qd','particle_q','particle_qd']:getattr(state,k).assign(z[k])
h.feedback.assign(z['feedback']);h.target.assign(z['target']);h.time=float(z['time']);i=int(np.argmin(abs(seq['time']-(h.time+.01))));h.set_targets(seq['targets'][i]);h.advance(.01,dt=.000125)
r=dict(checkpoint_time=float(z['time']),next_time=float(seq['time'][i]),max_body_position_difference_m=float(np.max(abs(h.state.body_q.numpy()[:,:3]-seq['body_q'][i,:,:3]))),max_body_quaternion_component_difference=float(np.max(abs(h.state.body_q.numpy()[:,3:]-seq['body_q'][i,:,3:]))),max_particle_difference_m=float(np.max(abs(h.state.particle_q.numpy()[:h.real_count]-seq['particles'][i]))));(OUT/(tag+'_resume_probe.json')).write_text(json.dumps(r,indent=2));print(r)
