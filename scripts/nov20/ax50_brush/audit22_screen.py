
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
ROOT=Path((str(_EVOLUTION_ROOT)));P=ROOT/'outputs/nov20/ax50_brush';O=P/'continue';sys.path.insert(0,str(ROOT/'scripts/nov20/ax50_brush'))
from integration import AX50WritingScene
import newton,warp as wp
from newton._src.sim.contact_kinematics import eval_rigid_contact_kinematics
R=Rot.from_euler('x',90,degrees=True);mount=np.array([0,0,.14]);r=AX50WritingScene(side='left',wrist='locked',position=mount,orientation=R.as_quat(),gravity=(0,0,0),iterations=60,brush_xml=P/'tongji_scene_source.xml');r.restore_prepared_hand(np.load(O/'opposed_axis_y_v2.npz'));q=r.state.joint_q.numpy();c=r.raw_contacts;dist=wp.zeros(c.rigid_contact_max,dtype=float,device=r.device);labels=r.model.shape_label;sb=r.model.shape_body.numpy();pts=r.state.particle_q.numpy();bones=np.asarray(r.pads['bone_names'])[r.pads['anchor_bone']];index=pts[np.flatnonzero(bones=='link_2_3')+r.pad_offset];rows=[]
for x in np.arange(.01,.0501,.001):
 for z in np.arange(-.015,.0151,.001):
  center=R.apply([x,.05,z])+mount;rad=np.linalg.norm(index[:,:2]-center[:2],axis=1)-.005
  sg=float(rad.min()-.00005)
  if not -.00025<sg<.0008:continue
  q[r.brush_q_start:r.brush_q_start+7]=np.r_[center,[0,0,0,1]];q[r.brush_hinge_q]=0;r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);r.collision.collide(r.state,c);eval_rigid_contact_kinematics(r.model,r.state,c,out_distance=dist);n=int(c.rigid_contact_count.numpy()[0]);a=c.rigid_contact_shape0.numpy()[:n];b=c.rigid_contact_shape1.numpy()[:n];ds=dist.numpy()[:n];contact=[]
  for aa,bb,dd in zip(a,b,ds):
   if aa<0 or bb<0:continue
   ha=0<=sb[aa]<r.hand_body_count;hb=0<=sb[bb]<r.hand_body_count
   if ha==hb:continue
   other=bb if ha else aa
   if sb[other]<r.brush_root_body:continue
   contact.append([labels[aa],labels[bb],float(dd)*1000])
  worst=min([e[2] for e in contact],default=1.);thumb=min([e[2] for e in contact if 'link_1_2' in (e[0]+e[1]) and 'brush_shaft' in e[:2]],default=10.)
  rows.append(dict(x=float(x),z=float(z),softgap_mm=sg*1000,worst_mm=worst,thumbgap_mm=thumb,contacts=contact))
rows.sort(key=lambda a: max(0,-a['worst_mm']-.05)*100+abs(a['thumbgap_mm'])+abs(a['softgap_mm']))
(O/'audit22_screen.json').write_text(json.dumps(rows,indent=2));print('CANDIDATES',len(rows));print(json.dumps(rows[:5],indent=2))
