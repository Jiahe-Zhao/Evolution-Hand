import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
for name,color,label in [('run15','#167d9a','dt = 1/240 s'),('run16','#c65d32','dt = 1/480 s')]:
 p=base/name/'trace.json'
 if not p.exists():continue
 rows=json.loads(p.read_text());t=[x['t'] for x in rows]
 axes[0].plot(t,[x['rise_m']*1000 for x in rows],color=color,label=label)
 axes[1].plot(t,[min(50,x['relative_slip_m']*1000) for x in rows],color=color)
 axes[2].plot(t,[sum(x['normal_force_N'][1:]) for x in rows],color=color)
axes[0].set_ylabel('Tool rise (mm)');axes[0].legend();axes[1].set_ylabel('Relative slip (mm)');axes[1].axhline(5,color='grey',linestyle='--',label='5 mm threshold');axes[1].set_ylim(0,50);axes[2].set_ylabel('Long-finger normal force (N)');axes[2].set_xlabel('Simulation time (s)')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Free-tool validation: timestep sensitivity\n80 g, unchanged friction; contact stiffness 1500 N/m')
fig.savefig(base/'timestep_comparison.png',dpi=170)
