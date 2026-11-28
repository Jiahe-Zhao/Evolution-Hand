
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
def evaluate(tag):
 o=OUT/'pour21';z=np.load(o/(tag+'.npz'));a=z['a'];xyz=z['xyz'];loc=z['loc'];ex=z['ex'];on,rim=z['flags'].transpose(1,0,2);end=ex[-1];inside=(abs(loc[-1,:,0])<.018)&(abs(loc[-1,:,1])<.016)&(loc[-1,:,2]>0)&(loc[-1,:,2]<.019);bc=np.array([-.066,-.057,.012]);inb=(np.linalg.norm(xyz[:,:,:2]-bc[:2],axis=2)<.058)&(xyz[:,:,2]>=.011)&(xyz[:,:,2]<.036);bowl=inb[-1]&end;lost=~(bowl|inside);mx=a[:,1].argmax();gt=np.flatnonzero(a[mx:,1]>5);up=mx+gt[-1]+1 if len(gt) else mx;exitidx=np.argmax(ex,axis=0);ids=np.flatnonzero(end);height=loc[exitidx[ids],ids,2];stable=np.linalg.norm(xyz[-240:,bowl]-xyz[-1,bowl],axis=2).max(initial=0)
 ok=9<=bowl.sum()<=11 and lost.sum()<=1 and not np.any(inside&end) and up+240<len(a) and not np.any(ex[up:]!=ex[up-1]) and a[:,4].max()==0 and a[:,6].max()==0 and a[:,5].min()>=2 and stable<.001 and (height>.016).all()
 return locals()
if __name__=='__main__':
 import sys
 for t in sys.argv[1:]:
  D=evaluate(t);print(t,D['ok'],'in',D['bowl'].sum(),'out',D['lost'].sum(),'left',D['inside'].sum(),'up',D['up']/480,'tilts',D['a'][:,1].max(),D['a'][-1,1],'ext hand',D['a'][:,[6,4]].max(0))
