
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np
from scipy.spatial.transform import Rotation as R
def snapshot(h):
 s=h.contacts;n=int(s.soft_contact_count.numpy()[0]);sb=h.model.shape_body.numpy();sh=s.soft_contact_shape.numpy()[:n];ids=np.flatnonzero(sb[sh]==h.item)
 f=h.out_f.numpy();nn=s.soft_contact_normal.numpy();dep=h.out_depth.numpy();mu=h.vbd.body_particle_contact_material_mu.numpy();ke=h.vbd.body_particle_contact_penalty_k.numpy();ii=s.soft_contact_indices.numpy();bb=s.soft_contact_barycentric.numpy();pv=h.state.particle_qd.numpy();pp=h.state.particle_q.numpy();bq=h.state.body_q.numpy();bv=h.state.body_qd.numpy();com=bq[h.item,:3]+R.from_quat(bq[h.item,3:]).apply(h.model.body_com.numpy()[h.item]);out=[]
 for j in ids:
  if np.linalg.norm(f[j])<1e-7:continue
  k=int(np.argmax(bb[j]));pid=ii[j,k];region=int(h.regions[pid]) if 0<=pid<h.real_count else -1
  valid=ii[j]>=0;pos=(pp[ii[j,valid]]*bb[j,valid,None]).sum(0);vel=(pv[ii[j,valid]]*bb[j,valid,None]).sum(0);rel=vel-bv[h.item,:3]-np.cross(bv[h.item,3:],pos-com);normal=nn[j];fn=float(f[j]@normal);ft=f[j]-normal*fn;cap=float(mu[j]*ke[j]*max(0,dep[j]));slip=rel-normal*(rel@normal)
  out.append([j,region,mu[j],ke[j],dep[j],fn,np.linalg.norm(ft),cap,np.linalg.norm(slip),*pos,*normal,*(-f[j])])
 return out
