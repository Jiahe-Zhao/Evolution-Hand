
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import fit_grasp as f
from scipy.optimize import differential_evolution
import numpy as np,json
sm,sd=f.distance_model(f.fingers,f.thumb);q0=sd.qpos.copy()
hids=[i for i in range(sm.ngeom) if sm.geom(i).name.startswith("hammer_") and (sm.geom_contype[i] or sm.geom_conaffinity[i])]
links=[f.model.body(int(f.model.geom_bodyid[i])).name for i in range(246)]
best=[1e9,None]
def objective(x):
 sd.qpos[:]=q0;sd.qpos[:3]+=np.array([x[0],x[1],0]);f.mj.mj_forward(sm,sd)
 ds=np.full(5,.12);worst=.12
 for i in range(246):
  name=links[i];digit=next((k for k in range(1,6) if f"link_{k}_" in name),0)
  for j in hids:
   d=f.mj.mj_geomDistance(sm,sd,i,j,.012,None);worst=min(worst,d)
   if digit:ds[digit-1]=min(ds[digit-1],d)
 score=1000*max(0,-worst+.00003)+sum(np.sort(np.maximum(ds,0))[:3])*100
 if score<best[0]:
  best[:]=[score,x.copy()];print(score,x,worst,ds,flush=True)
 return score
r=differential_evolution(objective,[(-.025,.025),(-.035,.035)],popsize=6,maxiter=15,tol=.01,polish=False,seed=7)
p=json.loads((f.OUT/"grasp_plan.json").read_text());p["hammer_in_hand"]=(f.HAMMER_IN_HAND+[r.x[0],r.x[1],0]).tolist();p["search_score"]=r.fun
(f.OUT/"grasp_clearance_plan.json").write_text(json.dumps(p,indent=2));print("DONE",r.fun,r.x,flush=True)
