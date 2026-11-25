"""Recorded force-control audit; never substitutes commanded force for measured force."""
from pathlib import Path
import csv,json,sys,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07');tag=sys.argv[1]
with (O/f'{tag}_trace.csv').open() as f:rows=list(csv.DictReader(f))
control=json.loads((O/f'{tag}_result.json').read_text())['force_control'];band=control['band_N'];abort=control['abort_sample_N']
a=lambda k:np.array([float(r[k]) for r in rows]);t=a('time_s');f0=a('thumb_terminal_N');f1=a('index_terminal_N');active=np.array([r['phase'] in ['force_limited_search','lift_20mm','lift_hold'] for r in rows]);lift=np.array([r['phase'] in ['lift_20mm','lift_hold'] for r in rows]);start=lambda x:float(t[np.flatnonzero(x)[0]]) if x.any() else None
r=dict(scope='100 Hz measured sample audit',thumb_first_contact_s=start(f0>1e-6),index_first_contact_s=start(f1>1e-6),dual_first_contact_s=start((f0>1e-6)&(f1>1e-6)),lift_start_recorded_s=start(lift),max_thumb_N=float(np.nanmax(f0)),max_index_N=float(np.nanmax(f1)),control=control,samples_over_band_upper=int(((np.maximum(f0,f1)>band[1])&active).sum()),samples_over_abort=int(((np.maximum(f0,f1)>abort)&active).sum()))
(O/f'{tag}_force_audit.json').write_text(json.dumps(r,indent=2))
fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
ax[0].plot(t,f0,label='thumb measured');ax[0].plot(t,f1,label='index measured');ax[0].axhspan(*band,alpha=.15,color='green',label='control band');ax[0].axhline(abort,color='r',ls='--',label='abort threshold');ax[0].set_ylabel('normal force (N)');ax[0].legend()
for k in ['thumb','index']:ax[1].plot(t,a(k+'_search_mm'),label=k)
ax[1].set_ylabel('reference approach (mm)');ax[1].legend();ax[2].plot(t,a('lift_mm'),label='target lift');ax[2].axhline(20,color='r',ls='--');ax[2].set_ylabel('lift (mm)');ax[2].set_xlabel('simulation time (s)')
for x in ax:x.grid(alpha=.3)
fig.tight_layout();fig.savefig(O/f'{tag}_force_curves.png',dpi=140);plt.close(fig);print(json.dumps(r))
