"""Audit recorded samples and plot contacts; does not rerun physics."""
from pathlib import Path
import csv,json,sys,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07');tag=sys.argv[1]
with (OUT/f'{tag}_trace.csv').open() as f:rows=list(csv.DictReader(f))
raw_rows=rows;rows=[r for r in rows if r['finite']=='True'];invalid=[r for r in raw_rows if r['finite']!='True']
a=lambda k:np.array([float(r[k]) for r in rows]);t=a('time_s');thumb=a('thumb_terminal_N');index=a('index_terminal_N');other=a('other_hand_item_N');surface=a('object_support_N');lift=a('lift_mm');neighbor=np.maximum.accumulate(np.maximum(a('neighbor1_displacement_mm'),a('neighbor2_displacement_mm')));tray=a('hand_tray_N')
ok=(thumb>1e-6)&(index>1e-6)&(other<=1e-6)&(surface<=1e-6)&(lift>=20)&(neighbor<=5);best=0.;start=None
for i,v in enumerate(ok):
 if v:
  if start is None:start=t[i]
  best=max(best,t[i]-start)
 else:start=None
r=dict(scope='100 Hz force-proxy audit; original acceptance uses geometric contacts; not a per-substep certification',tag=tag,max_lift_mm=float(lift.max()),max_neighbor_displacement_mm=float(neighbor.max()),max_hand_tray_N=float(tray.max()),max_thumb_N=float(thumb.max()),max_index_N=float(index.max()),dual_contact_samples=int(((thumb>1e-6)&(index>1e-6)).sum()),sampled_qualified_hold_s=float(best),sampled_criteria_met=bool(best>=.3 and not invalid),all_sampled_finite=not invalid,invalid_frames=len(invalid),first_invalid_time_s=float(invalid[0]['time_s']) if invalid else None)
(OUT/f'{tag}_audit.json').write_text(json.dumps(r,indent=2))
fig,ax=plt.subplots(4,1,figsize=(10,8),sharex=True)
ax[0].plot(t,lift);ax[0].axhline(20,color='r',ls='--');ax[0].set_ylabel('lift (mm)')
for v,label in [(thumb,'thumb'),(index,'index'),(other,'other hand')]:ax[1].plot(t,v,label=label)
ax[1].legend();ax[1].set_ylabel('target contact (N)');ax[2].plot(t,tray,label='hand-tray');ax[2].plot(t,surface,label='target-support');ax[2].legend();ax[2].set_ylabel('contact (N)');ax[3].plot(t,neighbor);ax[3].axhline(5,color='r',ls='--');ax[3].set_ylabel('neighbor max (mm)');ax[3].set_xlabel('simulation time (s)')
for x in ax:x.grid(alpha=.3)
fig.tight_layout();fig.savefig(OUT/f'{tag}_curves.png',dpi=140);plt.close(fig);print(json.dumps(r),flush=True)
