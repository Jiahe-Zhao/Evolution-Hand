
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,json,csv
from scipy.spatial.transform import Rotation as R
p=Path(__file__).parent;o=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));c=json.loads((p/'candidates/pinch_candidate_v4.json').read_text());cp=np.load(o/'jitter_source_state_20261127.npz');out={}
for tag in ['inward_0.25','inward_0.5','inward_1.0']:
 z=np.load(o/(tag+'.npz'),allow_pickle=True);names=[str(x) for x in z['body_names']];ids=[next(i for i,n in enumerate(names) if n.endswith(end)) for end in ['/link_1_2','/link_2_3','/item']];b=z['body_q'];b0=cp['body_q'];tips=[];starts=[]
 for j,k in enumerate(ids[:2]):tips.append(b[:,k,:3]+R.from_quat(b[:,k,3:]).apply(np.tile(c['local_points'][j],(len(b),1))));starts.append(b0[k,:3]+R.from_quat(b0[k,3:]).apply(c['local_points'][j]))
 axis=np.array(starts[0])-starts[1];axis/=np.linalg.norm(axis)
 # positive index towards thumb. Thumb outward increases span.
 proj=[(t-s)@axis*1000 for t,s in zip(tips,starts)];stone=(b[:,ids[2],:3]-b0[ids[2],:3])@axis*1000;span=(tips[0]-tips[1])@axis*1000;span0=(starts[0]-starts[1])@axis*1000
 a=np.genfromtxt(o/(tag+'_trace.csv'),names=True,delimiter=',');both=(a['thumb_mean_N']>.005)&(a['index_mean_N']>.005);runs=[];n=0
 for x in both:
  if x:n+=1
  elif n:runs.append(n*.005);n=0
 if n:runs.append(n*.005)
 samples=[]
 for t in [.005,.05,.1,.2,.3,.35,.4,.5]:
  i=int(np.argmin(abs(a['time_s']-t)));samples.append(dict(t=float(a['time_s'][i]),thumb_outward_mm=float(proj[0][i]),index_inward_mm=float(proj[1][i]),tip_span_change_mm=float(span[i]-span0),stone_along_pinch_mm=float(stone[i]),thumb_mean_N=float(a['thumb_mean_N'][i]),index_mean_N=float(a['index_mean_N'][i])))
 out[tag]=dict(pinch_axis=axis.tolist(),initial_reference_span_mm=float(span0),final_span_change_mm=float(span[-1]-span0),max_continuous_dual_contact_s=max(runs,default=0),thumb_only_fraction=float(np.mean((a['thumb_mean_N']>.005)&(a['index_mean_N']<=.005))),index_only_fraction=float(np.mean((a['index_mean_N']>.005)&(a['thumb_mean_N']<=.005))),samples=samples)
(o/'inward_cause_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
