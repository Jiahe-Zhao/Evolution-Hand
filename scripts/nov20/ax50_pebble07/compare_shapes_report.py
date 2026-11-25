"""Matched-shape comparison from recorded Newton trajectories, no resimulation."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/zjh/Evolution_PC'); O=ROOT/'outputs/nov20/ax50_pebble07'
summary={'scope':'same fresh-start targets, dt, joint caps, friction and acceptance; target shape only changed, inertia recomputed at same mass','variants':{}}
fig,axs=plt.subplots(3,1,figsize=(10,10),sharex=True)
zs=[]
for variant,col in [('original','#77513a'),('topwide','#168e8a')]:
 tag='shape_'+variant+'_v1'
 r=json.loads((O/(tag+'_result.json')).read_text())
 with (O/(tag+'_trace.csv')).open() as f: rows=list(csv.DictReader(f))
 a=lambda name:np.array([float(x[name]) for x in rows])
 t=a('time_s');lift=a('lift_mm');neighbor=np.maximum(a('neighbor1_displacement_mm'),a('neighbor2_displacement_mm'))
 z=np.load(O/(tag+'.npz'));zs.append(z)
 names=[str(n).rsplit('/',1)[-1] for n in z['body_names']]
 ids=np.flatnonzero(np.array([x['phase']=='lift_30mm' for x in rows]))
 hand={}
 if len(ids):
  for name in ['link_1_2','link_2_3']:
   j=names.index(name); hand[name]=float((z['body_q'][-1,j,2]-z['body_q'][max(0,ids[0]-1),j,2])*1000)
 conclusion='failed: target lift below 20 mm' if max(lift)<20 else 'requires original geometric-contact/hold audit'
 summary['variants'][variant]=dict(stop=r['stop'],recorded_s=r['time'],max_lift_mm=float(max(lift)),final_lift_mm=float(lift[-1]),max_neighbor_displacement_mm=float(max(neighbor)),thumb_peak_N=float(max(a('thumb_terminal_N'))),index_peak_N=float(max(a('index_terminal_N'))),max_hand_tray_N=float(max(a('hand_tray_N'))),actual_terminal_z_lift_mm=hand,finite=r['finite'],acceptance_passed=False if max(lift)<20 else None,conclusion=conclusion,object_xml=r['object_xml'])
 axs[0].plot(t,lift,color=col,label=variant)
 axs[1].plot(t,neighbor,color=col,label=variant)
 axs[2].plot(t,a('thumb_terminal_N'),color=col,label=variant+' thumb')
 axs[2].plot(t,a('index_terminal_N'),'--',color=col,label=variant+' index')
n=min(len(zs[0]['targets']),len(zs[1]['targets']))
summary['same_targets_common_prefix']=bool(np.array_equal(zs[0]['targets'][:n],zs[1]['targets'][:n]))
summary['target_max_difference']=float(np.max(np.abs(zs[0]['targets'][:n]-zs[1]['targets'][:n])))
summary['common_target_samples']=n
axs[0].axhline(20,color='gray',ls=':',label='20 mm criterion')
axs[1].axhline(5,color='gray',ls=':',label='5 mm limit')
for ax,label in zip(axs,['Target lift (mm)','Max neighbor displacement (mm)','Sampled normal force (N)']):
 ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(loc='upper left');ax.axvline(3.85,color='gray',alpha=.3)
axs[-1].set_xlabel('Simulation time (s)')
fig.suptitle('AX50 pebble shape A/B | Same clamp and lift commands\nContact forces sampled at 100 Hz; acceptance is unchanged')
fig.tight_layout();fig.savefig(O/'shape_comparison_curves.png',dpi=160)
review_path=O/'shape_comparison_review.json'
if review_path.exists():
 import hashlib
 review=json.loads(review_path.read_text())
 if review.get('result_sha256')==hashlib.sha256((O/'shape_topwide_v1_result.json').read_bytes()).hexdigest():
  summary['variants']['topwide'].update(acceptance_passed=review['acceptance_passed'],conclusion=review['conclusion'])
  summary['contact_geometry_review']=review
(O/'shape_comparison_result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
