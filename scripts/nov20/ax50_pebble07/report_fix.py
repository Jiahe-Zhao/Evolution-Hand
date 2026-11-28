
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
p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));summary={};data={};sub={}
tags=['components_base_1128','components_kd0_1128','components_frozen_1128','components_fix_1128','components_fix_high_1128']
for tag in tags:
 if not (p/(tag+'_substeps.npz')).exists():continue
 a=np.genfromtxt(p/(tag+'_trace.csv'),delimiter=',',names=True,encoding='utf8');z=np.load(p/(tag+'_substeps.npz'));b=z['data'];t=(np.arange(len(b))+1)*float(z['dt']);m=(a['time_s']>=.65)&(a['time_s']<=.8);n=(t>=.65)&(t<=.8);r=json.loads((p/(tag+'_result.json')).read_text());positive=b[:,3]>0
 summary[tag]=dict(stop=r['stop'],duration=float(a['time_s'][-1]),hold_mean_N=float(a['normal_N'][m].mean()),hold_std_N=float(a['normal_N'][m].std()),hold_substep_std_N=float(b[n,0].std()),hold_elastic_mean_N=float(b[n,1].mean()),hold_damping_mean_N=float(b[n,2].mean()),max_5ms_N=float(a['normal_N'].max()),max_substep_N=float(b[:,0].max()),max_soft_overlap_mm=float(b[:,4].max()),max_other_hand_N=float(a['other_hand_N'].max()),max_fixture_motion_mm=float(a['specimen_move_mm'].max()),final_normal_N=float(a['normal_N'][-1]),final_slide_mm=float(a['actual_tip_slide_mm'][-1]),mu_range=[float((b[positive,5]/b[positive,3]).min()),float((b[positive,5]/b[positive,3]).max())],kd_range=[float((b[positive,6]/b[positive,3]).min()),float((b[positive,6]/b[positive,3]).max())])
 data[tag]=a;sub[tag]=(t,b)
fig,ax=plt.subplots(3,2,figsize=(13,11))
for tag,label,color in [('components_base_1128','Before 0.4 mm','C3'),('components_fix_1128','After 0.4 mm','C0'),('components_fix_high_1128','After 0.7 mm','C2')]:
 if tag not in data:continue
 a=data[tag];t,b=sub[tag];ax[0,0].plot(a['time_s'],a['normal_N'],label=label,color=color)
 if 'fix' in tag:ax[0,1].plot(a['time_s'],a['normal_N'],label=label,color=color)
 ax[1,0].plot(t,b[:,4],label=label,color=color,lw=.5);mask=(t>=.65)&(t<=.7);ax[1,1].semilogy(t[mask],np.maximum(b[mask,0],1e-7),label=label,color=color,lw=.6)
 ax[2,0].plot(a['time_s'],a['actual_tip_slide_mm'],label=label,color=color);ax[2,1].plot(a['time_s'],a['tangent_N'],label=label,color=color)
for aa,title in zip(ax.flat,['Normal reaction, 5 ms average (N)','After: normal reaction detail (N)','Soft contact overlap (mm)','Unfiltered substep force during hold (N; log scale)','Tangential reference travel (mm)','Sliding-axis reaction, 5 ms average (N)']):aa.set_title(title,fontsize=10);aa.set_xlabel('Time (s)');aa.grid(alpha=.25);aa.legend(fontsize=8)
fig.suptitle('AX50 contact correction: identical friction, tissue and contact damping');fig.tight_layout();fig.savefig(p/'coupling_fix_comparison_1128.png',dpi=160)
f,axes=plt.subplots(2,2,figsize=(12,8))
for row,tag in enumerate(['components_base_1128','components_fix_1128']):
 if tag not in sub:continue
 t,b=sub[tag];mask=(t>=.65)&(t<=.7)
 for i,label in [(0,'total'),(1,'elastic penalty'),(2,'contact damping')]:axes[row,0].plot(t[mask],b[mask,i],label=label,lw=.7)
 axes[row,0].set_title(('Before' if row==0 else 'After')+' / unfiltered force components (N)');axes[row,0].legend();axes[row,0].grid(alpha=.25)
 a=data[tag];axes[row,1].plot(t,b[:,0],lw=.4,label='contact normal');axes[row,1].plot(t,-b[:,7],lw=.5,label='distal anchor component (not all hand reactions)');axes[row,1].set_title(('Before' if row==0 else 'After')+' / contact and distal anchor (N)');axes[row,1].legend(fontsize=8);axes[row,1].grid(alpha=.25)
f.tight_layout();f.savefig(p/'coupling_components_1128.png',dpi=160)
(p/'coupling_fix_summary_1128.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
