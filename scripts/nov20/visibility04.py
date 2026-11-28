
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import sys
o=OUT/'leaf04';EYE=np.array([-.121,.009,.35]);T=np.array([-.121,.009,.036])
def vis(tag,grid=(40,56),stride=1):
 m=mj.MjModel.from_xml_path(str(o/(tag+'.xml')));d=mj.MjData(m);D=np.load(o/(tag+'.npz'));q=D['q'];nx,ny=grid
 gx,gy=np.meshgrid(((np.arange(nx)+.5)/nx*2-1)*.011,((np.arange(ny)+.5)/ny*2-1)*.016);vec=np.c_[gx.ravel(),gy.ravel(),np.full(nx*ny,T[2]-EYE[2])];vec/=np.linalg.norm(vec,axis=1)[:,None];vec=vec.ravel();hit=np.empty(nx*ny,np.int32);ds=np.empty(nx*ny);food=m.geom('food').id;ids=[i for i in range(m.ngeom) if m.geom(i).name.startswith('handgeom')];left=m.geom('blade0').id;right=m.geom('blade1').id;out=[]
 for i in range(0,len(q),stride):
  d.qpos[:]=q[i];mj.mj_forward(m,d);mj.mj_multiRay(m,d,EYE,vec,None,True,-1,hit,ds,None,len(ds),.5);out.append([i/480,np.mean(hit==food),np.isin(hit,ids).mean(),np.mean(hit==left),np.mean(hit==right)])
 return np.array(out)
if __name__=='__main__':
 for k in sys.argv[1:]:
  z=vis(k);np.savetxt(o/(k+'_visibility.csv'),z,delimiter=',',header='time_s,visible,occluded_hand,occluded_leaf1,occluded_leaf2',comments='',fmt='%.7f');print(k,z[::240],flush=True)
