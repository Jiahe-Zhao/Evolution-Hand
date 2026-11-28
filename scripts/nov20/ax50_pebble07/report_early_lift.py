
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));P=Path(__file__).parent
orientation=json.loads((P/'candidates/pinch_candidate_v4.json').read_text())['orientation']
fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);report={}
for tag,label,col in [('early_lift_hold','Hold on tray','#ba6452'),('early_lift_probe','2 mm lift probe','#008f9a')]:
 r=json.loads((O/(tag+'_result.json')).read_text());a=np.genfromtxt(O/(tag+'_trace.csv'),delimiter=',',names=True,dtype=None,encoding='utf8');z=np.load(O/(tag+'.npz'),allow_pickle=True)
 trigger=r['trigger_s']
 if trigger is None:report[tag]=dict(trigger=None,stop=r['stop']);continue
 t=a['time_s'];j=int(np.argmin(abs(t-trigger)));tw=t-trigger;wrist=Rotation.from_quat(orientation).apply(z['joint_q'][:,:3])[:,2]
 post=(tw>=.2-1e-8)&(tw<=.4+1e-8);pre=(tw>=-.15)&(tw<=0)
 out=dict(trigger_s=trigger,stop=r['stop'],wrist_rise_mm=float((wrist[-1]-wrist[j])*1000),stone_rise_mm=float((a['stone_z'][-1]-a['stone_z'][j])*1000),micro_hold_off_tray_fraction=float(np.mean(a['object_support_N'][post]<=1e-5)) if post.any() else None,max_soft_reaction_N=r['max_soft_reaction_N'],max_native_depth_mm=r['max_native_depth_mm'])
 rel=np.column_stack([a[k] for k in ['relative_x','relative_y','relative_z']])
 for name,mask in [('pre',pre),('micro_hold',post)]:
  if mask.sum()<3:continue
  A=np.column_stack([np.ones(mask.sum()),t[mask]-t[mask][0]]);res=rel[mask]-A@np.linalg.lstsq(A,rel[mask],rcond=None)[0]
  out[name+'_relative_detrended_rms_mm']=float(np.sqrt(np.mean(np.sum(res*res,axis=1)))*1000)
 report[tag]=out
 axs[0,1].plot(tw,a['object_support_N'],color=col,label=label)
 axs[1,0].plot(tw,(rel[:,2]-rel[j,2])*1000,color=col,label=label)
 if tag=='early_lift_probe':
  axs[0,0].plot(tw,(wrist-wrist[j])*1000,label='Actual wrist');axs[0,0].plot(tw,(a['stone_z']-a['stone_z'][j])*1000,label='Actual stone');axs[0,0].plot(tw,a['command_lift_mm'],ls='--',label='Wrist command')
  axs[1,1].plot(tw,a['thumb_mean_N'],label='Thumb');axs[1,1].plot(tw,a['index_mean_N'],label='Index')
for ax in axs.ravel():ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlabel('Time from contact trigger (s)')
axs[0,0].set_ylabel('Vertical displacement (mm)');axs[0,1].set_ylabel('Stone-tray contact force (N)');axs[1,0].set_ylabel('Stone relative to fingers, z change (mm)');axs[1,1].set_ylabel('Mean contact force per 5 ms (N)')
fig.suptitle('AX50 early-lift diagnostic: recorded-state continuation, not task acceptance')
fig.savefig(O/'early_lift_comparison.png',dpi=150)
(O/'early_lift_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
