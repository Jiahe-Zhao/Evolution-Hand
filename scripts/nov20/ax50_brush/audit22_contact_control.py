
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
def static_audit(center):
 place(*center);rc=screen(*center);nc=int(c.soft_contact_count.numpy()[0]);sh=c.soft_contact_shape.numpy()[:nc];idx=c.soft_contact_indices.numpy()[:nc];bary=c.soft_contact_barycentric.numpy()[:nc];normal=c.soft_contact_normal.numpy()[:nc];bodypoint=c.soft_contact_body_pos.numpy()[:nc];selected=(sh==labels.index('brush_shaft'))&(idx[:,0]>=r.pad_offset)&(idx[:,0]<r.bridge_offset);p=np.sum(basep[np.maximum(idx,0)]*bary[:,:,None],axis=1);bq=r.state.body_q.numpy()[r.brush_root_body];bp=Rot.from_quat(bq[3:]).apply(bodypoint)+bq[:3];rad=np.where(idx>=0,r.model.particle_radius.numpy()[np.maximum(idx,0)],0).max(1);dep=rad+r.model.shape_margin.numpy()[np.maximum(sh,0)]-np.sum(normal*(p-bp),axis=1);si=np.flatnonzero(selected)[np.argmax(dep[selected])];print('STATIC',{'rigid_min_mm':min([v[2] for v in rc],default=None),'soft_max_depth_mm':float(dep[si]*1000),'soft_count':int(selected.sum())},flush=True);return p[si],-normal[si],dep[si],rc

def make_preload(center,force=.3):
 ip,inn,dep,rc=static_audit(center);q0=r.state.joint_q.numpy().copy();tip=r.body_names.index('link_1_2');index=r.body_names.index('link_2_3');bq=r.state.body_q.numpy();il=Rot.from_quat(bq[index,3:]).inv().apply(ip-bq[index,:3]);nt=Rot.from_quat(bq[tip,3:]).apply(nl);grad={}
 for group,bid,cp,normal in [(names,tip,point,nt),(['arch_2','digit_2_flex_1','digit_2_flex_2','digit_2_flex_3'],index,il,inn)]:
  bp=r.state.body_q.numpy()[bid];p0=Rot.from_quat(bp[3:]).apply(cp)+bp[:3]
  for name in group:
   j=r.joint_names.index(name);q=q0.copy();q[j]+=.0005;r.state.joint_q.assign(q);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state);bp=r.state.body_q.numpy()[bid];p1=Rot.from_quat(bp[3:]).apply(cp)+bp[:3];grad[name]=float((p1-p0)@normal/.0005)
  r.state.joint_q.assign(q0);newton.eval_fk(r.model,r.state.joint_q,r.state.joint_qd,r.state)
 pre={n:float(np.clip(baset[r.joint_names.index(n)]+v*force/4,r.model.joint_limit_lower.numpy()[r.joint_names.index(n)],r.model.joint_limit_upper.numpy()[r.joint_names.index(n)])-baset[r.joint_names.index(n)]) for n,v in grad.items()};print('PRELOAD',pre,'NORMALS',nt,inn,flush=True);return pre
