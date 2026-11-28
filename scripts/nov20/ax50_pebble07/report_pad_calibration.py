
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));out={}
fig,axs=plt.subplots(3,2,figsize=(13,11))
for tag,label,color in [('calibration_thumb_v2','Thumb / 0.7 mm','C0'),('calibration_index_v2','Index / 0.7 mm (stopped)','C3'),('calibration_index_low','Index / 0.4 mm','C2')]:
 a=np.genfromtxt(p/(tag+'_trace.csv'),delimiter=',',names=True,encoding='utf8');r=json.loads((p/(tag+'_result.json')).read_text());hold=(a['time_s']>=.65)&(a['time_s']<=.8);sl=(a['time_s']>=.85)&(a['time_s']<=1.15)&(a['normal_N']>.02);on=np.flatnonzero(a['normal_N']>.02);zero=a['actual_tip_press_mm'][on[0]] if len(on) else 0
 def stat(key,mask,fn=np.mean):return float(fn(a[key][mask])) if mask.any() else None
 sl &= abs(np.gradient(a['actual_tip_slide_mm'],a['time_s']))>1
 out[tag]={'stop':r['stop'],'duration_s':float(a['time_s'][-1]),'hold_normal_mean_N':stat('normal_N',hold),'hold_normal_std_N':stat('normal_N',hold,np.std),'hold_reference_press_mm':stat('actual_tip_press_mm',hold),'sliding_normal_mean_N':stat('normal_N',sl),'sliding_tangent_abs_mean_N':float(np.mean(abs(a['tangent_N'][sl]))) if sl.any() else None,'sliding_single_axis_force_ratio_median':float(np.median(abs(a['tangent_N'][sl])/a['normal_N'][sl])) if sl.any() else None,'max_normal_N':float(max(a['normal_N'])),'max_rigid_overlap_mm':float(max(a['rigid_depth_mm'])),'max_soft_peak_N':float(max(a['soft_peak_N'])),'max_other_hand_N':float(max(a['other_hand_N'])),'max_specimen_motion_mm':float(max(a['specimen_move_mm'])),'end_normal_N':float(a['normal_N'][-1]),'end_slide_mm':float(a['actual_tip_slide_mm'][-1])}
 for ax,x,y in [(axs[0,0],a['time_s'],a['normal_N']),(axs[0,1],a['actual_tip_press_mm']-zero,a['normal_N']),(axs[1,0],a['time_s'],a['tangent_N']),(axs[1,1],a['time_s'],a['actual_tip_slide_mm']),(axs[2,0],a['time_s'],a['rigid_depth_mm']),(axs[2,1],a['time_s'],a['active_count'])]:ax.plot(x,y,label=label,color=color,lw=1)
for ax,title in zip(axs.flat,['Normal reaction (N), 5 ms average','Reaction vs body reference travel (NOT tissue strain)','Signed sliding-axis reaction (N)','Tangential body reference travel (mm)','Rigid collision overlap (mm, NOT soft indentation)','Active contact constraints (substep mean)']):
 ax.set_title(title,fontsize=10);ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlabel('Time (s)')
axs[0,1].set_xlabel('Travel after contact onset (mm)')
fig.suptitle('AX50 fixed-plane baseline: friction and material settings unchanged',fontsize=13);fig.tight_layout();fig.savefig(p/'pad_calibration_curves.png',dpi=160)
(p/'pad_calibration_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
