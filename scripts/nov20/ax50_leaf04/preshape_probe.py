
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from leaf_runtime import *
import json
p=json.loads((OUT_DIR/'plan.json').read_text());offset=np.array([0,.12,0]);h=LeafRuntime(side='left',wrist='six_axis',position=np.array(p['position'])+offset,orientation=p['orientation'],gravity=(0,0,-9.81));base=np.r_[np.zeros(6),p['poses'][0],np.zeros(5)];rows=[]
for i in range(70):
 t=i*.01;u=min(t/.5,1);h.set_targets(base*u*u*(3-2*u));h.advance(.01,dt=.000125);v=h.state.particle_qd.numpy();q=h.state.joint_q.numpy();finite=bool(np.isfinite(v).all() and np.isfinite(q).all());c=h.contacts;n=int(c.soft_contact_count.numpy()[0]);ss=c.soft_contact_shape.numpy()[:n];ii=c.soft_contact_indices.numpy()[:n];bb=c.soft_contact_barycentric.numpy()[:n];ff=h.out_f.numpy()[:n];dd=h.out_depth.numpy()[:n];groups={}
 for j in np.flatnonzero(np.linalg.norm(ff,axis=1)>.01):
  pid=int(ii[j,np.argmax(bb[j])]);reg=h.region_names[h.regions[pid]] if 0<=pid<h.real_count and h.regions[pid]>=0 else 'other';lab=reg+' / '+str(h.model.shape_label[ss[j]]);g=groups.setdefault(lab,[0.,0.,0]);g[0]+=float(np.linalg.norm(ff[j]));g[1]=max(g[1],float(dd[j]));g[2]+=1
 row=dict(time=h.time,finite=finite,max_soft_speed=float(np.nanmax(np.linalg.norm(v,axis=1))),top=sorted(groups.items(),key=lambda x:x[1][0],reverse=True)[:5]);rows.append(row)
 if i%10==0:print(row,flush=True)
 if not finite or row['max_soft_speed']>20:break
(OUT_DIR/'preshape_corrected.json').write_text(json.dumps(rows,indent=2))
if rows[-1]['finite']:
 pts=h.state.particle_q.numpy()[:h.real_count]-offset
 for name,c,half in [('base0',[-.065,-.001,.055],[.003,.0025,.020]),('base1',[-.065,.020,.055],[.003,.0025,.020]),('earth',[-.113,.009,.031],[.05,.035,.003])]:
  b=abs(pts-c)-half;dist=np.linalg.norm(np.maximum(b,0),axis=1)+np.minimum(np.max(b,axis=1),0);ids=np.flatnonzero(dist<.0005);print('TARGET_CLEARANCE',name,float(dist.min()),len(ids),sorted(set(h.region_names[h.regions[j]] if h.regions[j]>=0 else 'bridge' for j in ids)),flush=True)
 np.savez_compressed(OUT_DIR/'preshape_corrected_checkpoint.npz',**{k:getattr(h.state,k).numpy() for k in ['joint_q','joint_qd','body_q','body_qd','particle_q','particle_qd']},body_names=h.body_names_all)
