
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys,numpy as np,json,time
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush';O=P/'continue';sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
import newton,warp as wp
from newton._src.sim.contact_kinematics import eval_rigid_contact_kinematics
r=AX50WritingScene(side='left',wrist='locked',position=(0,0,.14),gravity=(0,0,0),iterations=50,brush_xml=P/'tongji_scene_source.xml')
z=np.load(P/'prepared_hand_locked.npz');r.restore_prepared_hand(z);q=r.state.joint_q.numpy();q[r.brush_q_start:r.brush_q_start+7]=[1,1,1,0,0,0,1];q[r.brush_hinge_q]=0;r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);r.rigid.reset(r.state,flags=0);r.vbd.reset(r.state,flags=0);r.collision.reset_contact_matching()
for key,value in vars(r.state).items():
 dst=getattr(r.next,key,None)
 if isinstance(value,wp.array) and isinstance(dst,wp.array):wp.copy(dst,value)
r._advance_graph=None;t0=r.target.numpy().copy();t1=t0.copy()
for n in ['arch_4','arch_5','digit_4_flex_1','digit_4_flex_2','digit_4_flex_3','digit_5_flex_1','digit_5_flex_2','digit_5_flex_3']:t1[r.joint_names.index(n)]=0
for k in range(50):
 u=min(1,(k+1)/35);r.target.assign(t0+(t1-t0)*(u*u*(3-2*u)));r.advance(.01)
 if k%10==0:print('PREP',k,flush=True)
 if not np.isfinite(r.state.particle_q.numpy()).all():raise RuntimeError('preparation nonfinite')
np.savez(O/'audit22_three_prepared.npz',joint_q=r.state.joint_q.numpy()[:r.hand_dof_count],target=r.target.numpy()[:r.hand_dof_count],particle_q=r.state.particle_q.numpy(),action_names=r.action_names)
baseq=r.state.joint_q.numpy().copy();basep=r.state.particle_q.numpy().copy();baset=r.target.numpy().copy();labels=r.model.shape_label;sb=r.model.shape_body.numpy();c=r.raw_contacts;dist=wp.zeros(c.rigid_contact_max,dtype=float,device=r.device);pl=(r.pads['vertices']-r.pads['rest_body_matrices'][0,:3,3])@r.pads['rest_body_matrices'][0,:3,:3];ix=np.flatnonzero(pl[:,2]>.063)+r.pad_offset

def place(x,y,z=.125):
 q=baseq.copy();q[r.brush_q_start:r.brush_q_start+7]=[x,y,z,0,0,0,1];q[r.brush_hinge_q]=0;r.state.joint_q.assign(q);r.state.joint_qd.zero_();r.state.particle_q.assign(basep);r.state.particle_qd.zero_();newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);r.state.body_qd.zero_();r.target.assign(baset);r.rigid.reset(r.state,flags=0);r.vbd.reset(r.state,flags=0);r.collision.reset_contact_matching();r.feedback.zero_();r._advance_graph=None;r.time=0.
 for key,value in vars(r.state).items():
  dst=getattr(r.next,key,None)
  if isinstance(value,wp.array) and isinstance(dst,wp.array):wp.copy(dst,value)

def screen(x,y,z=.125):
 q=baseq.copy();q[r.brush_q_start:r.brush_q_start+7]=[x,y,z,0,0,0,1];q[r.brush_hinge_q]=0;r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);r.collision.collide(r.state,c);eval_rigid_contact_kinematics(r.model,r.state,c,out_distance=dist);n=int(c.rigid_contact_count.numpy()[0]);a=c.rigid_contact_shape0.numpy()[:n];b=c.rigid_contact_shape1.numpy()[:n];dd=dist.numpy()[:n];out=[]
 for aa,bb,d in zip(a,b,dd):
  if aa<0 or bb<0:continue
  ha=0<=sb[aa]<r.hand_body_count;hb=0<=sb[bb]<r.hand_body_count
  if ha==hb or sb[bb if ha else aa]<r.brush_root_body:continue
  out.append([labels[aa],labels[bb],float(d*1000)])
 return out

def scan():
 out=[]
 for x in np.arange(.078,.087,.00025):
  for y in np.arange(.039,.053,.00025):
   gap=float(np.linalg.norm(basep[ix,:2]-[x,y],axis=1).min()-.00505)
   if not -.00015<gap<.0005:continue
   con=screen(x,y);worst=min([a[2] for a in con],default=1);th=min([a[2] for a in con if 'link_1_2' in str(a[:2])],default=10);out.append(dict(x=x,y=y,gap_mm=gap*1000,worst_mm=worst,thumb_mm=th,contacts=con))
 out.sort(key=lambda a: max(0,-a['worst_mm']-.05)*100+abs(a['thumb_mm'])+abs(a['gap_mm']));(O/'audit22_three_screen.json').write_text(json.dumps(out,indent=2));print('SCAN',len(out),[{k:v for k,v in a.items() if k!='contacts'} for a in out[:8]],flush=True);return out
print('LAB_READY',flush=True)
