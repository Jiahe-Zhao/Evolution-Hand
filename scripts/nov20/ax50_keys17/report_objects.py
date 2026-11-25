from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_keys17');a=np.genfromtxt(p/'object_probe.csv',delimiter=',',names=True)
f,ax=plt.subplots(figsize=(10,4))
for k in range(1,5):ax.plot(a['time_s'],a[f'key{k}_m']*1000,label=f'K{k}')
for y,label in [(2,'trigger'),(.5,'released'),(4,'nominal stop')]:ax.axhline(y,ls='--',lw=.8,color='gray');ax.text(2.4,y,label,ha='right',va='bottom')
ax.set(xlabel='Time (s)',ylabel='Travel (mm)',title='Object-only diagnostic: 0.72 N step per key (not hand acceptance)');ax.legend();f.tight_layout();f.savefig(p/'object_probe.png',dpi=160)
