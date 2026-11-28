
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial import cKDTree
import sys,json
o=OUT/'hao14';tag=sys.argv[1];z=np.load(o/(tag+'.npz'));a=z['a'];curves=[np.array(x) for x in json.loads((o/(tag+'_curves.json')).read_text())]
ink=(a[:,9]>.01)&np.isfinite(a[:,7]);held=(a[:,10:12]>.01).all(1);ok=ink&held;metrics=[]
for sid,curve in enumerate(curves):
 ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];ss=np.linspace(0,ar[-1],5001);dense=np.array([np.interp(ss,ar,curve[:,j]) for j in range(2)]).T;tree=cKDTree(dense)
 idx=np.flatnonzero(ok&(a[:,12]==sid)&np.isin(a[:,13],[0,1]));err,ix=tree.query(a[idx,7:9]);rms=np.sqrt(np.mean(err**2))*1000 if len(err) else 999
 ints=sorted((min(ss[x],ss[y]),max(ss[x],ss[y])) for k,(x,y) in enumerate(zip(ix,ix[1:])) if idx[k+1]-idx[k]==1 and max(err[k],err[k+1])<=.003);end=-1;tot=0
 for x,y in ints:tot+=max(0,y-max(end,x));end=max(end,y)
 metrics.append(dict(stroke=sid+1,coverage_pct=100*tot/ar[-1],rms_mm=rms,length_mm=ar[-1]*1000,ink_steps=len(idx),grip_min=a[(a[:,12]==sid)&(a[:,13]==1),10:12].min(0).tolist()))
m=dict(strokes=metrics,ink_in_travel_steps=int((ink&(a[:,13]==3)).sum()),unheld_ink_steps=int((ink&~held).sum()),max_force_N=float(a[:,9].max()),total_steps=len(a),time_s=float(a[-1,0]))
(o/(tag+'_metrics.json')).write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
