from contact_audit import *
import json,csv
p=json.loads((OUT_DIR/'plan_tipclear.json').read_text());z=np.load(OUT_DIR/'hand_v7_checkpoint_239.npz');h=AuditedLeafRuntime(side='left',wrist='six_axis',position=p['position'],orientation=p['orientation'],gravity=(0,0,-9.81));h.vbd.leaf_contact_ke=500.;h.vbd.leaf_contact_kd=.05
for state in [h.state,h.next]:
 for key in ['joint_q','joint_qd','particle_q','particle_qd','body_q','body_qd']:getattr(state,key).assign(z[key])
h.time=float(z['time']);rows=[];base=z['target'];amp=float(os.environ.get('LIFT_AMPLITUDE','.001'));tag=os.environ.get('LIFT_TAG','hold_lift1mm')
for i in range(100):
 t=i*.01;target=base.copy();u=np.clip((t-.1)/.8,0,1);target[1]+=amp*u*u*(3-2*u);h.set_targets(target);h.advance(.01,dt=.0000625);rows.append([h.time,*h.state.joint_q.numpy()[h.leaf_q_np],*h.audit_values()])
 if i%10==0:print(round(h.time,3),h.audit_values(),flush=True)
a=np.asarray(rows);valid=a[:,4]==a[:,3];idx=np.flatnonzero(valid);groups=np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else [];best=max(groups,key=len) if groups else [];r=dict(scope='checkpoint-resume action diagnostic; not full task acceptance',lift_m=amp,qualified_hold_s=len(best)*.01,interval=[float(a[best[0],0]-.01),float(a[best[-1],0])] if len(best) else [],min_tip_N=a[:,5:7].min(0).tolist(),max_other_N=float(a[:,7].max()),final_leaf_deg=np.rad2deg(a[-1,1:3]).tolist());(OUT_DIR/(tag+'.json')).write_text(json.dumps(r,indent=2));np.savetxt(OUT_DIR/(tag+'.csv'),a,delimiter=',',header=','.join(['time_s','leaf0_rad','leaf1_rad']+AUDIT_COLUMNS),comments='');print(r,flush=True)
