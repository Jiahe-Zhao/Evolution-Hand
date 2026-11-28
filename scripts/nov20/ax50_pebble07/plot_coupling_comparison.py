
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'))
names=['coupling_components_kdNone_guarded_125us','coupling_components_kd0.5_guarded_125us'];labels=['Original contact damping: 55 N s/m','Target-only damping: 0.5 N s/m'];colors=['#c6504b','#168b8c']
fig,axs=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
res=[]
for name,label,col in zip(names,labels,colors):
 a=np.genfromtxt(O/(name+'_trace.csv'),delimiter=',',names=True);c=np.genfromtxt(O/(name+'_components.csv'),delimiter=',',names=True);r=json.loads((O/(name+'_result.json')).read_text());res.append(r)
 axs[0,0].plot(a['time_s']*1000,a['soft_peak_N'],label=label,color=col)
 axs[0,1].plot(a['time_s']*1000,np.sqrt(a['v0']**2+a['v1']**2+a['v2']**2),color=col)
 axs[1,0].plot(c['time_s']*1000,c['damping_N'],color=col)
axs[0,0].set(ylabel='Soft reaction peak per 1 ms bin (N)',xlabel='Replay elapsed (ms)',yscale='symlog');axs[0,0].legend(fontsize=8)
axs[0,1].set(ylabel='Stone speed, sampled at 1 kHz (m/s)',xlabel='Replay elapsed (ms)')
axs[1,0].set(ylabel='Normal damping contribution (N)',xlabel='Replay elapsed (ms)',yscale='symlog')
x=np.arange(2);elastic=[r['components_at_peak_total'][1] for r in res];damp=[r['components_at_peak_total'][2] for r in res]
axs[1,1].bar(x,elastic,label='Elastic',color='#ddaa45');axs[1,1].bar(x,damp,bottom=elastic,label='Damping',color='#9671ad');axs[1,1].set(xticks=x,xticklabels=['55 N s/m','0.5 N s/m'],ylabel='At peak summed normal force (N)');axs[1,1].legend()
for ax in axs.ravel():ax.grid(alpha=.2)
fig.suptitle('AX50 pebble contact: 250 ms diagnostic, reset velocities\nSame geometry/controller; no grasp acceptance claimed',fontsize=13)
fig.savefig(O/'coupling_control_comparison.png',dpi=160)
