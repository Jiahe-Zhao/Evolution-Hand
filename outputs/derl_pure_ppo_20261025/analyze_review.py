"""Summarize paired checkpoint rollouts without pooling repeated falls as trials."""
import json,sys
from pathlib import Path
import numpy as np
b=Path(sys.argv[1]);rows=json.loads((b/'summary.json').read_text());output=[]
for r in rows:
 p=b/(r['label']+'_trace.npz')
 if not p.exists():continue
 a=np.load(p);t=a['t'];falls=a['falls']>0;first=np.where(falls.any(0),t[falls.argmax(0)],np.nan)
 x=dict(label=r['label'],successes=r['successes'],num_envs=r['num_envs'],ever_held_3s=r['ever_held_3s'],unique_falls=int(falls[-1].sum()),first_fall_median_s=float(np.nanmedian(first)) if falls.any() else None,max_digit_force_N=r['diagnostics']['max_digit_force_N'],force_exceedances=r['diagnostics']['force_exceedances'],case0_success=r['passed'][0],case0_first_fall_s=float(first[0]) if np.isfinite(first[0]) else None,snapshots=[])
 for sec in [5,10,15,17.95,18.5,20,22,25]:
  i=np.argmin(abs(t-sec));x['snapshots'].append(dict(t=float(t[i]),held=int(a['held'][i].sum()),fallen=int(falls[i].sum()),height_median=float(np.median(a['object'][i,:,2])),speed_median=float(np.median(a['speed'][i])),wrist_mean=a['wrist'][i].mean(0).tolist(),action_wrist_mean=a['action'][i,:,21:].mean(0).tolist(),thumb_force_median=float(np.median(np.linalg.norm(a['force'][i,:,0],axis=-1))),reward_mean=float(a['reward'][i].mean())))
 output.append(x)
(b/'analysis.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
