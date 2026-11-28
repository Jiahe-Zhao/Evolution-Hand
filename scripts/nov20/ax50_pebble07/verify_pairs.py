
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pebble_runtime import *
from scipy.spatial.transform import Rotation
h=PebbleRuntime(side='left',wrist='six_axis',position=(.25,.25,.35),gravity=(0,0,-9.81))
def read():
 c=h.rigid.mjw_data.contact;n=min(int(h.rigid.mjw_data.nacon.numpy()[0]),h.rigid.mjw_data.naconmax);g=c.geom.numpy()[:n];f=c.friction.numpy()[:n];s=c.solref.numpy()[:n];imp=c.solimp.numpy()[:n];dim=c.dim.numpy()[:n];dist=c.dist.numpy()[:n];m=h.rigid.mj_model;rows=[];arr=h.pair_override_arrays;active=arr['active'].numpy();expected={k:v.numpy() for k,v in arr.items()};ok=True
 for i,(a,b) in enumerate(g):
  k=int(a)*m.ngeom+int(b)
  if not active[k] or dist[i]>0:continue
  match=bool(np.allclose(f[i],expected['friction'][k],atol=1e-7) and np.allclose(s[i],expected['solref'][k],atol=1e-7) and np.allclose(imp[i],expected['solimp'][k],atol=1e-7) and dim[i]==expected['dim'][k]);ok &= match
  rows.append(dict(geoms=[m.geom(int(a)).name,m.geom(int(b)).name],friction=f[i].tolist(),solref=s[i].tolist(),solimp=imp[i].tolist(),dim=int(dim[i]),distance_m=float(dist[i]),match=match))
 return dict(rows=rows,all_match=ok,nonpositive_contacts=len(rows))
h.pair_overrides_enabled=False;h.advance(.002,dt=.000125);before=read()
h.pair_overrides_enabled=True;h._advance_graph=None;h.advance(.002,dt=.000125);after=read()
# Diagnostic state only: rotate one free stone so its mesh, rather than its foot,
# makes contact with the unchanged tray. Original XML and task reset are untouched.
q=h.state.joint_q.numpy();k=h.free_joint_rows[0]['q_start'];q[k+3:k+7]=Rotation.from_euler('y',.5).as_quat();h.state.joint_q.assign(q);newton.eval_fk(h.model,h.state.joint_q,h.state.joint_qd,h.state);h.advance(.002,dt=.000125);stone=read()
foot_ok=after['all_match'] and any(any('foot' in n for n in x['geoms']) and any('tray' in n for n in x['geoms']) for x in after['rows']);stone_ok=stone['all_match'] and any(any('stone0' in n for n in x['geoms']) and any('tray' in n for n in x['geoms']) for x in stone['rows'])
r=dict(scope='actual GPU contact parameter test; rotated-stone phase is a diagnostic reset, not grasp acceptance',mapping=h.pair_override_info,before=before,after=after,rotated_stone=stone,foot_tray_pass=foot_ok,stone_tray_pass=stone_ok,passed=foot_ok and stone_ok)
(OUT/'pair_verification.json').write_text(json.dumps(r,indent=2));print('PAIR_RESULT',r['passed'],foot_ok,stone_ok,flush=True)
raise SystemExit(0 if r['passed'] else 1)
