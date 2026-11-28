
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from card19 import *
def metrics(tag):
 f=np.load(out/(tag+'.npz'));x=f['data'];p=x[:,1:36].reshape(-1,5,7);init=f['initial'];delta=p[:,:,:3]-init[None,:,:3];d=np.linalg.norm(delta,axis=-1)*1000;rel=1000*(delta[-1,4,0]-delta[-1,3,0]);v=np.array([[xx,yy,z] for xx,y,z0 in [(-.027,.015,.00021),(-.025,.016,.0004),(.025,.016,.0004),(.027,.015,.00021)] for yy in (-y,y) for z in (-z0,z0)])
 corners=[]
 for i in range(5):
  pts=np.einsum('tij,vj->tvi',R.from_quat(p[:,i,3:]).as_matrix(),v)+p[:,i,None,:3];corners.append(float(1000*np.linalg.norm(pts-(R.from_quat(init[i,3:]).apply(v)+init[i,:3]),axis=2).max()))
 res=dict(final_x_mm=(delta[-1,:,0]*1000).tolist(),max_com_mm=d.max(0).tolist(),max_corner_mm=corners,relative_mm=float(rel),top_touched=bool((x[:,37]>0).any()),other_hand_steps=int((x[:,38]>0).sum()),card_edge_contact_steps=int((x[:,39]>0).sum()),final_hand_free=bool(np.all(x[-240:,37:39]==0)),duration=x[-1,0],success=bool(rel>=30 and delta[-1,4,0]>=.030 and max(corners[:4])<=5 and (x[:,37]>0).any() and not x[-240:,37:39].any()))
 return res
if __name__=='__main__':
 for tag in sys.argv[1:]:
  r=metrics(tag);print(tag,round(r['relative_mm'],2),np.round(r['max_corner_mm'][:4],2),r['other_hand_steps'],r['success'])
