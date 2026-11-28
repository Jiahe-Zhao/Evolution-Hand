
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np,sys
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07'));tag=sys.argv[1] if len(sys.argv)>1 else 'vector_fixed_grasp_50g_1128';a=np.genfromtxt(p/(tag+'_trace.csv'),delimiter=',',names=True,encoding='utf8');r=json.loads((p/(tag+'_result.json')).read_text());t=a['time_s'];trigger=r['trigger_s'];mask=(t>=trigger+.4) if trigger is not None else (t>=max(.2,t[-1]-.5));summary={}
for k in ['thumb_mean_N','index_mean_N','thumb_fz','index_soft_nz','index_soft_tz','index_soft_fz','index_rigid_fz','support_fz','soft_compression_max_mm','soft_overlap_mm','stone_speed_m_s']:
 summary[k]={'mean':float(a[k][mask].mean()),'min':float(a[k][mask].min()),'max':float(a[k][mask].max())}
r['evaluation_window']='after lift ramp' if trigger is not None else 'last 0.5 s; no lift triggered';r['window_summary']=summary;r['final_wrist_rise_mm']=float(a['wrist_rise_mm'][-1]);r['max_other_hand_N']=float(a['other_hand_item_N'].max());r['max_soft_peak_N']=float(a['soft_peak_N'].max());r['final_closure_mm']=[float(a['thumb_command_mm'][-1]),float(a['index_command_mm'][-1])]
z=np.load(p/(tag+'_friction.npz'),allow_pickle=True);values=[]
for s in z['soft']:
 b=np.asarray(s,dtype=float).reshape(-1,18)
 if len(b):values.extend(b[:,2].tolist())
r['soft_mu_range']=[min(values),max(values)] if values else []
fig,ax=plt.subplots(3,2,figsize=(13,11))
for k,l in [('thumb_mean_N','thumb'),('index_mean_N','index')]:ax[0,0].plot(t,a[k],label=l)
ax[0,0].plot(t,a['soft_peak_N'],alpha=.35,label='soft substep peak')
for k,l in [('lift_mm','stone rise'),('wrist_rise_mm','wrist rise')]:ax[0,1].plot(t,a[k],label=l)
ax[0,1].axhline(2,color='k',ls='--',label='2 mm target')
for k,l in [('thumb_fz','thumb'),('index_soft_fz','index soft'),('support_fz','tray')]:ax[1,0].plot(t,a[k],label=l)
ax[1,0].axhline(.4905,color='k',ls='--',label='weight 0.4905 N')
for k,l in [('index_soft_nz','index normal Z'),('index_soft_tz','index friction Z')]:ax[1,1].plot(t,a[k],label=l)
for k,l in [('soft_compression_max_mm','soft normal deformation'),('soft_overlap_mm','soft collision overlap'),('thumb_depth_mm','thumb rigid overlap'),('index_depth_mm','index rigid overlap')]:ax[2,0].plot(t,a[k],label=l)
for k,l in [('stone_x','X'),('stone_y','Y'),('stone_z','Z')]:ax[2,1].plot(t,(a[k]-a[k][0])*1000,label=l)
for aa,title in zip(ax.flat,['Finger normal loads (N, 5 ms averages)','Measured lift (mm)','Vertical forces on stone (N)','Index vertical force components (N)','Skin deformation vs collision overlap (mm)','Stone displacement from first recorded frame (mm)']):
 aa.set_title(title,fontsize=10);aa.set_xlabel('Time (s)');aa.grid(alpha=.25);aa.legend(fontsize=8)
 if trigger is not None:aa.axvline(trigger,color='gray',ls=':')
fig.suptitle('AX50 corrected contact / 50 g stone / '+r['stop']);fig.tight_layout();fig.savefig(p/(tag+'_curves.png'),dpi=160)
f,axes=plt.subplots(1,3,figsize=(15,4))
for j,xyz in enumerate(['x','y','z']):
 for group in ['thumb','index_soft','support']:axes[j].plot(t,a[group+'_f'+xyz],label=group)
 axes[j].set_title('Force on stone: '+xyz.upper()+' (N)');axes[j].set_xlabel('Time (s)');axes[j].grid(alpha=.25);axes[j].legend()
f.tight_layout();f.savefig(p/(tag+'_force_xyz.png'),dpi=160)
(p/(tag+'_summary.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

from scipy.spatial.transform import Rotation as R
poses=np.load(p/(tag+'.npz'),allow_pickle=True);bq=poses['body_q'];names=poses['body_names'];c=json.loads((Path(__file__).parent/'candidates/pinch_candidate_v4.json').read_text());points=[]
for k,body in enumerate(['link_1_2','link_2_3']):
 bid=next(i for i,n in enumerate(names) if str(n).endswith('/'+body));points.append(bq[:,bid,:3]+R.from_quat(bq[:,bid,3:]).apply(np.tile(c['local_points'][k],(len(bq),1))))
axis=points[0][0]-points[1][0];axis[2]=0;axis/=np.linalg.norm(axis)
f,aa=plt.subplots(1,2,figsize=(12,4))
for k,key in enumerate(['thumb','index']):
 actual=(points[k]-points[k][0])@(axis*(-1 if k==0 else 1))*1000
 r[key+'_actual_inward_final_mm']=float(actual[-1]);r[key+'_actual_inward_peak_mm']=float(actual.max())
 aa[k].plot(t,a[key+'_command_mm'],label='command');aa[k].plot(t,actual,label='body-attached reference point');aa[k].set_title(key+' inward travel (mm)');aa[k].set_xlabel('Time (s)');aa[k].legend();aa[k].grid(alpha=.25)
f.tight_layout();f.savefig(p/(tag+'_closure.png'),dpi=160)
if 'potential_up_N' in a.dtype.names:r['potential_up_final_window_mean_N']=float(a['potential_up_N'][mask].mean());r['potential_up_final_window_min_N']=float(a['potential_up_N'][mask].min())
(p/(tag+'_summary.json')).write_text(json.dumps(r,indent=2));print('actual closure',[(k,r[k+'_actual_inward_final_mm']) for k in ['thumb','index']])
