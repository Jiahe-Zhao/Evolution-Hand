from keys_runtime import *
import json,sys
from scipy.spatial.transform import Rotation as R
import damped_adapter as da
from wrist_runtime import wrist_pd
da.control_pd_subset=wrist_pd
class Isolated(da.AX50BrushRuntime):
 def _add_brush(self,b): self.prop_body_count=0
out=ROOT/'outputs/nov20/ax50_keys17';plan=json.loads((out/'left_plan.json').read_text())
h=Isolated(side='left',wrist='six_axis',position=(np.array(plan['position'])+[0,0,.04]).tolist(),orientation=plan['orientation'],gravity=(0,0,-9.81))
base=np.r_[np.zeros(6),plan['base'],np.zeros(5)]; rows=[];poses=[];particles=[];times=[]
for i in range(220):
 t=i*.01;target=base*(.5-.5*np.cos(np.pi*min(t/.5,1)))
 if t>=1.:
  r=.5-.5*np.cos(np.pi*min((t-1.)/.6,1));target[0]=.005*r;target[3]=np.deg2rad(5)*r
 h.set_targets(target);h.advance(.01,dt=.000125)
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);idx=c.soft_contact_indices.numpy()[:n];bary=c.soft_contact_barycentric.numpy()[:n];sh=c.soft_contact_shape.numpy()[:n];normal=c.soft_contact_normal.numpy()[:n];local=c.soft_contact_body_pos.numpy()[:n];vel=c.soft_contact_body_vel.numpy()[:n]
 p=h.state.particle_q.numpy();pp=h.vbd.particle_q_prev.numpy();bq=h.state.body_q.numpy();old=h.next.body_q.numpy();sb=h.model.shape_body.numpy();f=h.out_f.numpy()[:n];depth=h.out_depth.numpy()[:n];ke=h.vbd.body_particle_contact_penalty_k.numpy()[:n];kd=h.vbd.body_particle_contact_material_kd.numpy()[:n]
 records=[]
 for j in range(n):
  if not str(h.model.shape_label[int(sh[j])]).endswith('/link_5_1/skin_5_1'):continue
  if not any(h.regions[k]==0 for k,w in zip(idx[j],bary[j]) if w>1e-5 and 0<=k<h.real_count):continue
  body=sb[sh[j]];rot=R.from_quat(bq[body,3:]);brot=R.from_quat(old[body,3:]);bx=rot.apply(local[j])+bq[body,:3];prev= brot.apply(local[j])+old[body,:3];x=np.sum(p[idx[j]]*bary[j,:,None],axis=0);xp=np.sum(pp[idx[j]]*bary[j,:,None],axis=0)
  vn=float(np.dot(normal[j],(x-xp-bx+prev)/.000125-rot.apply(vel[j])))
  records.append(dict(depth=float(depth[j]),ke=float(ke[j]),kd=float(kd[j]),vn=vn,elastic=float(max(depth[j],0)*ke[j]),damping=float(max(-vn,0)*kd[j]) if depth[j]>0 else 0.,force=f[j].tolist(),normal=normal[j].tolist(),point=x.tolist(),body_point=bx.tolist(),particle_ids=idx[j].tolist()))
 rows.append(dict(time=h.time,contacts=records,max_qd=float(np.max(abs(h.state.joint_qd.numpy()))),max_soft_v=float(np.max(np.linalg.norm(h.state.particle_qd.numpy(),axis=1))),q=h.state.joint_q.numpy().tolist(),qd=h.state.joint_qd.numpy().tolist(),target=target.tolist()))
 poses.append(bq);particles.append(p[:h.real_count]);times.append(h.time)
 if i%10==0:print(i,rows[-1]['max_qd'],len(records),flush=True)
 (out/'repair_pair_detail.json').write_text(json.dumps(rows))
np.savez_compressed(out/'repair_pair.npz',body_q=poses,particles=particles,time=times,body_names=h.body_names_all,tri=h.model.tri_indices.numpy(),real_count=h.real_count)
print('DONE',flush=True)
