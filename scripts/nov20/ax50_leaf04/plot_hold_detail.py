from pathlib import Path
import csv,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_leaf04');rows=[r for r in csv.DictReader((p/'hand_v8_trace.csv').open()) if 2.5<=float(r['time_s'])<=3.3];x=[float(r['time_s']) for r in rows];fig,ax=plt.subplots(figsize=(10,4.5))
for a,b,color,label in [('min_leaf0_index_N','max_leaf0_index_N','C0','Index distal'),('min_leaf1_middle_N','max_leaf1_middle_N','C1','Middle distal')]:
 lo=[float(r[a]) for r in rows];hi=[float(r[b]) for r in rows];ax.fill_between(x,lo,hi,alpha=.25,color=color,label=label+' min/max');ax.plot(x,lo,color=color,linewidth=.8);ax.plot(x,hi,color=color,linewidth=.8)
ax.plot(x,[float(r['max_unassigned_N']) for r in rows],color='C2',label='Other hand regions: maximum');ax.axhline(.01,color='r',linestyle='--',label='Required minimum: 0.01 N');ax.set(xlabel='Simulation time (s)',ylabel='Normal force (N)',title='AX50 04 hold: every physics substep, ranges in 10 ms windows');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(p/'hand_v8_hold_contacts.png',dpi=160)
