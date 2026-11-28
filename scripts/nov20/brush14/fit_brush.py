
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
import sys,os,json
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/brush14';OUT.mkdir(exist_ok=True)
fmt=lambda x:' '.join(map(str,x))
r=_evo_parse_xml(ROOT/'outputs/nov20/pen14/write.xml').getroot();w=r.find('worldbody');hand=w.find('body');w.remove(w.find("body[@name='pen']"))
for g in list(w.findall('geom')):w.remove(g)
for tag in ['contact','actuator']:
 el=r.find(tag)
 if el is not None:r.remove(el)
ob=E.SubElement(w,'body',name='brush',pos='-.10 .02 .085');E.SubElement(ob,'freejoint',name='brush_free')
E.SubElement(ob,'geom',name='shaft',type='cylinder',fromto='0 0 -.035 0 0 .095',size='.005',mass='.009',rgba='.49 .28 .105 1',friction='1.1 .004 .0001')
E.SubElement(ob,'geom',name='axis',type='capsule',fromto='0 0 -.035 0 0 .095',size='.00005',mass='.0000001',contype='0',conaffinity='0',rgba='0 0 0 0')
m=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));d=mj.MjData(m);seed=np.load(ROOT/'outputs/nov20/pen14/warm.npz')['q'][:27].copy();seed[15:19]=seed[11:15];seed[19:23]=seed[11:15];seed[23:27]=[0,1.1,1.3,1.2]
q=seed.copy();q[:3]=0
pads=[m.geom('handgeom'+str(i)).id for i in [20,40,60,80]];axis=m.geom('axis').id;shaft=m.geom('shaft').id
other=[i for i in range(m.ngeom) if i not in pads and i not in [axis,shaft]]
center=np.array([-.075,.02,.090]);rad=[np.array([0,1,0]),np.array([0,-1,0]),np.array([-.75,-.66,0]),np.array([-.9,.43,0])];heights=[.005,.005,-.008,-.020]
N=int(os.getenv('FINGERS','4'));params=np.r_[q[3:23],center];lo=np.r_[m.jnt_range[3:23,0],[-.13,-.05,.04]];hi=np.r_[m.jnt_range[3:23,1],[.02,.09,.18]]
params=np.clip(params,lo+1e-5,hi-1e-5)
def residual(x,depth):
 d.qpos[:27]=q;d.qpos[3:23]=x[:20];d.qpos[27:30]=x[20:];d.qpos[30:34]=[1,0,0,0];mj.mj_kinematics(m,d)
 z=np.zeros(6);ans=[]
 for k,i in enumerate(pads[:N]):
  mj.mj_geomDistance(m,d,i,axis,.25,z)
  target=x[20:]+rad[k]*(.005-depth)+np.array([0,0,heights[k]])
  ans.extend((z[:3]-target)*1000)
 for i in other:
  dist=mj.mj_geomDistance(m,d,i,shaft,.008,z)
  ans.append(min(0,(dist-.0004)*1000)*1.5)
 ans.extend(.15*(x[:20]-params[:20]));ans.extend((x[20:]-center)*2)
 return np.array(ans)
best=None
for k in range(int(os.getenv('TRIES','3'))):
 x=params.copy()
 if k:x[:20]+=np.random.default_rng(k).normal(0,.25,20)
 x=np.clip(x,lo+1e-5,hi-1e-5)
 sol=least_squares(residual,x,args=(0.,),bounds=(lo,hi),max_nfev=250,diff_step=1e-4,ftol=1e-7)
 cost=np.linalg.norm(residual(sol.x,0));print('fit',k,cost,'tips',residual(sol.x,0)[:3*N],flush=True)
 if best is None or cost<best[0]:best=(cost,sol.x.copy())
x=best[1];sol=least_squares(residual,x,args=(float(os.getenv('DEPTH','.0003')),),bounds=(lo,hi),max_nfev=120,diff_step=1e-4)
initial=q.copy();initial[3:23]=x[:20];control=initial.copy();control[3:23]=sol.x[:20]
ob.set('pos',fmt(x[20:]));ob.remove(ob.find("geom[@name='axis']"))
E.ElementTree(r).write(OUT/'pose.xml');np.savez(OUT/'pose.npz',q=initial,control=control,center=x[20:],cost=best[0]);print('center',x[20:],'cost',best[0])
