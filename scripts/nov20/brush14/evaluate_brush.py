
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1];D=np.load(P/(tag+'.npz'));a=D['a'];q=D['q'];meta=json.loads((P/(tag+'_model.json')).read_text());m=mj.MjModel.from_xml_path(str(P/(tag+'.xml')));d=mj.MjData(m);shaft=m.geom('shaft').id;pb=m.geom_bodyid[shaft];tips=[i for i in range(m.ngeom) if m.geom(i).name.endswith('_tip')];palm=m.body('link_palm').id
width=[];bend=[];lag=[];relative=[];tiploc=[];rest=None
for qq in q:
 d.qpos[:]=qq;mj.mj_forward(m,d);R=d.xmat[pb].reshape(3,3);local=(d.geom_xpos[tips]-d.xpos[pb])@R
 if rest is None:rest=local.copy()
 width.append(np.max(np.linalg.norm(local[:,None,:2]-local[None,:,:2],axis=-1)));bend.append(np.linalg.norm(local-rest,axis=1).max());lag.append((local-rest)[:,0].mean());relative.append((d.xpos[pb]-d.xpos[palm])@d.xmat[palm].reshape(3,3));tiploc.append(local)
width=np.array(width);bend=np.array(bend);lag=np.array(lag);relative=np.array(relative)
hover=(a[:,0]>.8)&(a[:,0]<1.15);press=(a[:,0]>2.7)&(a[:,0]<3.08);drag=(a[:,0]>3.5)&(a[:,0]<4.8);recover=a[:,0]>7.3;active=a[:,0]>.8
airphase=(a[:,1]>=5)|hover
load=a[:,10]>.0001
stats={'tag':tag,'backend':'Newton 1.6.1 SolverMuJoCo CPU','free_brush':True,'duration_s':float(a[-1,0]),'total_mass_kg':float(sum(m.body_mass[i] for i in range(pb,m.nbody))), 'paper_force_peak_N':float(a[:,10].max()),'paper_force_press_mean_N':float(a[press,10].mean()),'hand_z_hover_mean_N':float(a[hover,13].mean()),'hand_z_press_mean_N':float(a[press,13].mean()),'tip_span_hover_mm':float(width[hover].mean()*1000),'tip_span_press_mm':float(width[press].mean()*1000),'tip_span_peak_mm':float(width.max()*1000),'tip_bend_max_mm':float(bend.max()*1000),'drag_mean_tip_lag_mm':float(lag[drag].mean()*1000),'tip_bend_recovered_mm':float(bend[recover].mean()*1000) if recover.any() else None,'hover_ink_contact_count':int(a[airphase,25].sum()),'hover_force_peak_N':float(a[airphase,10].max()),'hand_paper_peak_N':float(a[:,22].max()),'shaft_paper_peak_N':float(a[:,23].max()),'hair_paper_max_overlap_mm':float(-a[:,24].min()*1000),'grip_contact_fraction_per_finger':((a[active,14:19]>.005).mean(axis=0)).tolist(),'at_least_3_fingers_fraction':float(((a[active,14:19]>.005).sum(axis=1)>=3).mean()),'shaft_tilt_max_deg':float(a[active,26].max()),'shaft_translation_in_palm_max_mm':float(np.linalg.norm(relative[active]-relative[np.flatnonzero(hover)[-1]],axis=1).max()*1000),'hand_normal_peak_N':float(a[:,14:19].max()),'model_sha256':hashlib.sha256((P/(tag+'.xml')).read_bytes()).hexdigest()}
if len(sys.argv)>2:
 air=np.load(P/(sys.argv[2]+'.npz'))['a'];selection=(air[:,0]>2.7)&(air[:,0]<3.08);delta=air[selection,13].mean()-a[press,13].mean();stats.update({'air_control_ink_count':int(air[:,25].sum()),'air_control_paper_force_peak_N':float(air[:,10].max()),'air_control_hand_z_press_mean_N':float(air[selection,13].mean()),'hand_transmitted_reaction_delta_N':float(delta),'reaction_balance_relative_error':float(abs(delta-a[press,10].mean())/max(a[press,10].mean(),1e-9))})
(P/(tag+'_metrics.json')).write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True)
axs[0].plot(a[:,0],a[:,10],label='Paper on brush Z');axs[0].plot(a[:,0],a[:,13],label='Hand on brush Z');axs[0].set_ylabel('Force (N)');axs[0].legend()
axs[1].plot(a[:,0],width*1000,label='Tip span');axs[1].plot(a[:,0],bend*1000,label='Max tip deflection');axs[1].plot(a[:,0],lag*1000,label='Mean X deflection');axs[1].set_ylabel('mm');axs[1].legend()
for j,name in enumerate(['Thumb','Index','Middle','Ring','Little']):axs[2].plot(a[:,0],a[:,14+j],label=name)
axs[2].set_ylabel('Grip normal (N)');axs[2].set_xlabel('Time (s)');axs[2].legend(ncol=5)
for ax in axs:
 for x in [1.2,2.5,3.1,5.1,6.1,7.1]:ax.axvline(x,color='grey',lw=.6,ls='--')
 ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(P/(tag+'_curves.png'),dpi=160)
np.savetxt(P/(tag+'_deformation.csv'),np.c_[a[:,0],width,bend,lag,relative],delimiter=',',header='time_s,tip_span_m,max_tip_deflection_m,mean_tip_x_deflection_m,shaft_in_palm_x_m,shaft_in_palm_y_m,shaft_in_palm_z_m',comments='')
