from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_leaf04');fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
for a,tag,label in zip(ax,['hold_original','hold_low_damping','hold_soft_penalty'],['Original: ke=11250, kd=0.5','Damping only: ke=11250, kd=0.05','Reduced contact penalty: ke=500, kd=0.05']):
 x=np.loadtxt(p/(tag+'.csv'),delimiter=',',skiprows=1);a.plot((x[:,0]-x[0,0])*1000,x[:,4],linewidth=.6);a.axhline(.01,color='r',linestyle='--',linewidth=.7);a.set_title(label,loc='left');a.set_ylabel('Normal force (N)');a.grid(alpha=.25)
ax[-1].set_xlabel('Time from checkpoint (ms)');fig.suptitle('Same saved physical state; every 62.5 us; diagnostic, not task acceptance');fig.tight_layout();fig.savefig(p/'contact_probe_comparison.png',dpi=150)
