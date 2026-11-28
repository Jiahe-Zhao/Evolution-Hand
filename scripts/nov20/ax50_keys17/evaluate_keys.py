
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_keys17'));tag=sys.argv[1]
r=list(csv.DictReader((p/(tag+'.csv')).open()));t=np.array([float(x['time']) for x in r]);v=np.array([[float(x[f'key{k}_m']) for k in range(1,5)] for x in r]);contacts=list(csv.DictReader((p/(tag+'_contacts.csv')).open()));result=dict(tag=tag,duration_s=float(t[-1]),peak_mm=(v.max(0)*1000).tolist(),last_mm=(v[-1]*1000).tolist(),contact_rows=len(contacts),sampling_s=.01)
recording=np.load(p/(tag+'.npz'));result['recorded_body_poses_finite']=bool(np.isfinite(recording['body_q']).all())
result['recorded_contact_forces_finite']=bool(all(np.isfinite(float(x[k])) for x in contacts for k in ['fx_N','fy_N','fz_N','magnitude_N']))
if (p/(tag+'_substeps.json')).exists():
 audit=json.loads((p/(tag+'_substeps.json')).read_text());peaks=np.array(audit['phase_peaks_m']);checks=[]
 for k in range(4):
  press=1+2*k;release=2+2*k;rs=[i for i,x in enumerate(r) if x['phase']==f'released_{k+1}'];phasecontacts=[x for x in contacts if int(x['key'])==k+1 and v[int(np.argmin(abs(t-float(x['time'])))),k]>=.002];expected=[];other=[]
  for x in phasecontacts:
   label=x['body_or_region'];ok=(f'link_{k+2}_' in label) or (k==0 and 'index_' in label and 'palm' not in label)
   (expected if ok else other).append(x)
  checks.append(dict(key=k+1,peak_press_mm=float(peaks[press,k]*1000),trigger=bool(peaks[press,k]>=.002),other_key_peak_mm=float(np.max(np.delete(peaks[press:release+1],k,axis=1))*1000),no_neighbor_trigger=bool(np.max(np.delete(peaks[press:release+1],k,axis=1))<.002),release_mm=float(v[rs[-1],k]*1000) if rs else None,released=bool(rs and v[rs[-1],k]<=.0005),assigned_finger_contact_rows=len(expected),other_contact_labels=sorted(set(x['body_or_region'] for x in other))))
 approach=v[(t>=4.9)&(t<=5.11)];result['extra_approach_sampled_peak_mm']=(approach.max(0)*1000).tolist() if len(approach) else None
 result.update(checks=checks,all_substep_peaks_mm=(peaks*1000).tolist(),nonfinite=audit['nonfinite_count'],passed=bool(result['recorded_body_poses_finite'] and result['recorded_contact_forces_finite'] and t[-1]>=6.49 and audit['nonfinite_count']==0 and all(c['trigger'] and c['released'] and c['no_neighbor_trigger'] and c['assigned_finger_contact_rows']>0 and not c['other_contact_labels'] for c in checks) and peaks[0].max()<.002 and peaks[9].max()<.002 and np.max(v)<.01 and (not len(approach) or approach.max()<.002)))
else:result['scope']='single finger diagnostic, 10ms sampling; not full four-finger acceptance'
fig,ax=plt.subplots(figsize=(9,4))
for k in range(4):ax.plot(t,v[:,k]*1000,label=f'Key {k+1}')
ax.axhline(2,c='black',ls='--',label='Trigger 2mm');ax.axhline(.5,c='grey',ls=':',label='Released 0.5mm');ax.set(xlabel='Time (s)',ylabel='Key travel (mm)',title=tag);ax.grid(alpha=.25);ax.legend(ncol=3);fig.tight_layout();fig.savefig(p/(tag+'_travel.png'),dpi=160)
(p/(tag+'_result.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
