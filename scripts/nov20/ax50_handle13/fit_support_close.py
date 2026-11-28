"""CPU candidate: one finger keeps support while the middle finger curls."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,sys,numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/ax50_handle13';source=sys.argv[1] if len(sys.argv)>1 else 'open_support_v2';meta=json.loads((OUT/(source+'_model.json')).read_text());cp=np.load(OUT/(source+'_checkpoint.npz'));seq=np.load(OUT/(source+'.npz'));rot=R.from_quat(meta['plan']['orientation']);mount=np.array(meta['mount']);p0=mount+rot.apply(seq['targets'][219,:3]);q0=np.array(meta['plan']['open_pose']);bar=cp['body_q'][-1];rb=R.from_quat(bar[3:]);center=bar[:3]+rb.apply([.064,0,0])
m=mj.MjModel.from_binary_path(str(ROOT/'outputs/nov20/ax50_keys17/left_ik.mjb'));d=mj.MjData(m);conv=json.loads((ROOT/'outputs/ax50_visualskin_v3_0_0/ax50_visualskin/assets/left/joint_convention.json').read_text())['canonical_to_native'];sign=np.array([conv[m.joint(i).name.split('_Joints_')[-1]] for i in range(16)]);ids=[next(i for i in range(m.nbody) if m.body(i).name.endswith('_'+n)) for n in ['link_3_2','link_4_2']];local0=[c['local_point'] for c in meta['plan']['contact_points']]
def fk(q,p,points):
 d.qpos[:]=0;d.qpos[:16]=q*sign;mj.mj_kinematics(m,d)
 pts=np.array([rot.apply(d.xpos[b]+d.xmat[b].reshape(3,3)@v-[0,0,.3])+p for b,v in zip(ids,points)])
 ns=np.array([rot.apply(d.xmat[b].reshape(3,3)[:,1]) for b in ids]);return pts,ns
initial,_=fk(q0,p0,local0);ys=rb.inv().apply(initial-center)[:,1]
poses=[q0.tolist()];positions=[p0.tolist()];errors=[None];locals_used=[local0];x=np.r_[p0,0.,0.,local0[1][2]]
for u in [.2,.4,.6,.8,1.]:
 def calc(x):
  q=q0.copy();q[5]=x[3];q[6]=1.1*u;q[7]=1.3*u;q[9]=x[4];pts,ns=fk(q,x[:3],[[0,.0044,local0[0][2]],[0,.0044,x[5]]]);ln=rb.inv().apply(ns);target=center+rb.apply(np.c_[-.003*np.tanh(ln[:,0]/.12),ys,-.003*np.tanh(ln[:,2]/.12)]);return q,pts,target
 prev=x.copy()
 def fun(x):
  q,pts,target=calc(x);return np.r_[((pts-target)*1000).ravel(),(x[:3]-prev[:3])*2,x[3:5]*.3,(x[5]-local0[1][2])*2]
 f=least_squares(fun,x,bounds=(np.r_[p0-.035,0.,0.,.001],np.r_[p0+.035,.3,.5,.025]),max_nfev=400);x=f.x;q,pts,target=calc(x);poses.append(q.tolist());positions.append(x[:3].tolist());errors.append(((pts-target)*1000).tolist());locals_used.append([[0,.0044,local0[0][2]],[0,.0044,float(x[5])]])
r=dict(source=source,source_time=float(cp['time']),bar_center=center.tolist(),bar_quaternion=bar[3:].tolist(),u=[0,.2,.4,.6,.8,1.],positions=positions,poses=poses,errors_mm=errors,local_contact_points=locals_used,scope='CPU contact-point candidate only; ring PIP/DIP remain open, middle curls only after support. No physical grasp verdict.')
(OUT/'support_close_plan.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
