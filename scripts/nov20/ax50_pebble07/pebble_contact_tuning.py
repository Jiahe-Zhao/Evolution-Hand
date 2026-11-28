"""Optional scene-local hand/target rigid contact time constant experiment."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np

def set_hand_stone_tau(h,tau):
 # Change numerical recovery time only. Friction, solimp, object pairs and VBD unchanged.
 sb=h.model.shape_body.numpy();gs=h.rigid.mjc_geom_to_newton_shape.numpy().reshape(-1)
 gb=np.array([sb[s] if s>=0 else -1 for s in gs]);n=len(gb)
 refs=h.pair_override_arrays['solref'].numpy();active=h.pair_override_arrays['active'].numpy().reshape(n,n)
 a=refs.reshape(n,n,2);changed=[]
 for i in range(n):
  for j in range(n):
   if active[i,j] and ((gb[i]==h.item and 0<=gb[j]<h.hand_body_count) or (gb[j]==h.item and 0<=gb[i]<h.hand_body_count)):
    changed.append((i,j,float(a[i,j,0])));a[i,j,0]=tau
 h.pair_override_arrays['solref'].assign(refs);h._advance_graph=None
 return dict(scope=__doc__,old_tau_s=sorted(set(v for _,_,v in changed)),new_tau_s=tau,directed_rules=len(changed))
