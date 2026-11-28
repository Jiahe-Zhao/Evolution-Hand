
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import newton,json
o=OUT/'leaf04';cfg=json.loads((OUT.parent.parent/'evolution_tasks/v5_newton/configs/forage.json').read_text())['source_contract']['config']
for idx in range(2):
 r=E.Element('mujoco',model='legacy_free_leaf_probe');w=E.SubElement(r,'worldbody');E.SubElement(w,'geom',name='table',type='box',size='.35 .35 .05',pos='0 0 .05',friction='.5 .001 .0001')
 for j,n in enumerate(['leaf_one_cfg','leaf_two_cfg']):
  g=cfg[n]['spawn'];b=E.SubElement(w,'body',name='leaf'+str(j),pos=f'{[-.018,.018][j]} .006 {[.1022,.1064][j]}');E.SubElement(b,'freejoint',name='free'+str(j));E.SubElement(b,'geom',type='box',name='blade'+str(j),size=' '.join(map(str,np.array(g['size'])/2)),mass=str(g['mass_props']['mass']),friction='.5 .001 .0001')
 p=o/f'legacy{idx}.xml';E.ElementTree(r).write(p);b=newton.ModelBuilder();b.add_mjcf(str(p),collapse_fixed_joints=False);m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True);c=m.control();base=s.joint_q.numpy().reshape(2,7)[:,:3].copy();rows=[]
 for k in range(720):
  s.clear_forces();f=np.zeros((m.body_count,6),np.float32)
  for bi,nm in enumerate(["leaf_one_cfg","leaf_two_cfg"]):f[bi,:3]=-cfg[nm]["spawn"]["mass_props"]["mass"]*(cfg[nm]["spawn"]["rigid_props"].get("linear_damping") or 0)*s.body_qd.numpy()[bi,:3]
  if 100<=k<133:f[idx,0]+=[-1,1][idx]*.65
  s.body_f.assign(f);v.step(s,n,c,None,1/480);s,n=n,s;rows.append([k/480,*((s.joint_q.numpy().reshape(2,7)[:,:3]-base)*1000).ravel()])
 np.savetxt(o/f'legacy{idx}.csv',rows,delimiter=',',header='s,leaf1_dx_mm,dy1,dz1,leaf2_dx_mm,dy2,dz2',comments='');print(idx,rows[-1])
