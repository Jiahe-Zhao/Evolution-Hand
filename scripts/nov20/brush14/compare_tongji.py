
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'))
tags=['tongji_hand111',sys.argv[1]]
out={};fig,axes=plt.subplots(3,1,figsize=(12,10))
for tag,label in zip(tags,['Before','After']):
 d=np.load(P/(tag+'.npz'));a=d['a'];t=a[:,0];write=a[:,1]==3
 # Compare actual rigid brush motion, not filtered controller signals or rendered fibers.
 filt=butter(3,5,fs=480,btype='highpass',output='sos')
 hp=sosfiltfilt(filt,a[:,5:8],axis=0)*1000
 tilt_hp=sosfiltfilt(filt,a[:,26])
 m=json.loads((P/(tag+'_writing_metrics.json')).read_text())
 out[tag]={'duration_s':float(t[-1]),'tilt_peak_deg':float(a[:,26].max()),'writing_tilt_rms_deg':float(np.sqrt(np.mean(a[write,26]**2))),'brush_position_above_5Hz_rms_mm':float(np.sqrt(np.mean(np.sum(hp[write]**2,axis=1)))),'brush_tilt_above_5Hz_rms_deg':float(np.sqrt(np.mean(tilt_hp[write]**2))),'passed_strokes':sum(v['pass'] for v in m['strokes']),'weighted_coverage_pct':float(np.average([v['coverage_pct'] for v in m['strokes']],weights=[v['target_length_mm'] for v in m['strokes']])),'max_stroke_rms_mm':max(v['contact_centroid_rms_mm'] or 0 for v in m['strokes']),'air_contact_count':m['air_contact_count'],'success':m['success']}
 axes[0].plot(t,a[:,26],label=label);axes[1].plot(t,np.linalg.norm(hp,axis=1),label=label,alpha=.8);axes[2].plot(t,a[:,10],label=label,alpha=.8)
for ax,title in zip(axes,['Actual brush tilt (deg)','Actual brush position >5 Hz (mm)','Measured paper reaction (N)']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
axes[-1].set_xlabel('Simulation time (s)');fig.tight_layout();fig.savefig(P/(tags[1]+'_comparison.png'),dpi=160)
(P/(tags[1]+'_comparison.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
