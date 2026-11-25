"""Diagnostic only: resume saved physical state and hold for 0.15 s."""
from leaf_runtime import *
import json,csv
p=json.loads((OUT_DIR/os.environ.get('PROBE_PLAN','plan_tipclear.json')).read_text());tag=os.environ.get('PROBE_TAG','hold_kd05');kd=float(os.environ.get('LEAF_KD','0.5'));dt=float(os.environ.get('PROBE_DT','0.0000625'));z=np.load(OUT_DIR/os.environ.get('PROBE_CHECKPOINT','hand_v6_checkpoint_239.npz'))
h=LeafRuntime(side='left',wrist='six_axis',position=p['position'],orientation=p['orientation'],gravity=(0,0,-9.81));h.vbd.leaf_contact_kd=kd
if os.environ.get("LEAF_KE"):h.vbd.leaf_contact_ke=float(os.environ["LEAF_KE"])
for state in [h.state,h.next]:
 for key in ['joint_q','joint_qd','particle_q','particle_qd','body_q','body_qd']:getattr(state,key).assign(z[key])
h.time=float(z['time']);h.set_targets(z['target']);sb=h.model.shape_body.numpy();rows=[];leaf_dof=int(h.leaf_dof_idx.numpy()[0])
for i in range(round(.15/dt)):
 h.step(dt)
 c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);sh=c.soft_contact_shape.numpy()[:n];force=h.out_f.numpy()[:n];normal=c.soft_contact_normal.numpy()[:n];depth=h.out_depth.numpy()[:n];q=h.state.joint_q.numpy();qd=h.state.joint_qd.numpy();sel=sb[sh]==h.leaf_bodies[0];ff=force[sel];nn=normal[sel];dep=depth[sel];fn=np.abs((ff*nn).sum(1));plane=np.array([np.sin(q[h.leaf_q_np[0]]),0,np.cos(q[h.leaf_q_np[0]])]);net=ff.sum(0)@plane if len(ff) else 0.;vals=h.vbd.body_particle_contact_material_ke.numpy()[:n][sel]
 rows.append([h.time,q[h.leaf_q_np[0]],qd[leaf_dof],len(ff),float(fn.sum()),float(net),float(dep.max()) if len(dep) else 0.,float(vals.max()) if len(vals) else 0.])
 if not np.isfinite(q).all():break
r=np.array(rows);out=dict(kd=kd,dt=dt,scope='state-resume diagnostic only; feedback starts at zero; not task acceptance',finite=bool(np.isfinite(r).all()),angle_range_deg=np.rad2deg([r[:,1].min(),r[:,1].max()]).tolist(),max_angular_speed=float(abs(r[:,2]).max()),mean_contact_count=float(r[:,3].mean()),max_contact_count=float(r[:,3].max()),mean_normal_N=float(r[:,4].mean()),max_normal_N=float(r[:,4].max()),zero_load_fraction=float((r[:,4]<.0005).mean()),mean_signed_plane_force_N=float(r[:,5].mean()),max_depth_mm=float(r[:,6].max()*1000),max_contact_ke=float(r[:,7].max()))
(OUT_DIR/(tag+'.json')).write_text(json.dumps(out,indent=2));np.savetxt(OUT_DIR/(tag+'.csv'),r,delimiter=',',header='time_s,leaf_rad,leaf_speed_rad_s,soft_contact_count,normal_sum_N,signed_plane_force_N,max_depth_m,max_ke',comments='');print(out,flush=True)
