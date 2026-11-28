
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
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1];D=np.load(P/(tag+'.npz'));a=D['a'];curves=[np.array(v) for v in json.loads((P/(tag+'_curves.json')).read_text())];ink=D['ink'];report=[];fig,ax=plt.subplots(figsize=(7,7))
for sid,curve in enumerate(curves):
 ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];arc=np.linspace(0,ar[-1],6001);dense=np.c_[[np.interp(arc,ar,curve[:,k]) for k in [0,1]]].T;tree=cKDTree(dense)
 selected=(a[:,27]==sid)&np.isfinite(a[:,30])&(a[:,10]>1e-4);held=(a[:,14:18]>.005).all(axis=1);ix=np.flatnonzero(selected);err,idx=tree.query(a[ix,30:32]);valid=held[ix]&(err<=.003);intervals=[]
 for k in range(len(ix)-1):
  i,j=ix[k:k+2];ds=abs(arc[idx[k+1]]-arc[idx[k]]);eu=np.linalg.norm(a[j,30:32]-a[i,30:32])
  if j==i+1 and valid[k] and valid[k+1] and a[i,1] in (2,3) and a[j,1] in (2,3) and ds<max(.0005,3*eu):intervals.append(sorted([arc[idx[k]],arc[idx[k+1]]]))
 end=-1.;length=0.
 for x,y in sorted(intervals):length+=max(0,y-max(x,end));end=max(end,y)
 rows=ink[np.isin(ink[:,0].astype(int),ix)] if len(ink) else np.empty((0,6));point_errors=tree.query(rows[:,2:4])[0] if len(rows) else np.array([])
 rms=float(np.sqrt(np.mean(err**2))*1000) if len(err) else None;coverage=100*length/ar[-1]
 result={'stroke':sid+1,'coverage_pct':float(coverage),'contact_centroid_rms_mm':rms,'all_contact_points_rms_mm':float(np.sqrt(np.mean(point_errors**2))*1000) if len(point_errors) else None,'target_length_mm':float(ar[-1]*1000),'load_samples':len(ix),'four_finger_contact_fraction':float(held[ix].mean()) if len(ix) else 0.,'pass':bool(coverage>=90 and rms is not None and rms<=3 and len(ix)>0 and held[ix].all())};report.append(result)
 ax.plot(curve[:,0]*1000,curve[:,1]*1000,'--',color='#c98a32',lw=1.5,label='Target strokes' if sid==0 else None)
 actual=np.full((len(a),2),np.nan);actual[ix]=a[ix,30:32];ax.plot(actual[:,0]*1000,actual[:,1]*1000,color='#153f68',lw=1.3,label='Actual contact centroid' if sid==0 else None)
 ax.text(curve[0,0]*1000,curve[0,1]*1000,str(sid+1),fontsize=9,color='#b77722')
if len(ink):ax.scatter(ink[::3,2]*1000,ink[::3,3]*1000,s=.15,c='#223333',alpha=.13,label='Physical bristle contacts')
status=json.loads((P/(tag+'_solver_status.json')).read_text()) if (P/(tag+'_solver_status.json')).exists() else {'failure':'missing solver audit'}
air=np.isin(a[:,1],[0,5,6]);success=all(v['pass'] for v in report) and a[air,25].sum()==0 and status['failure'] is None and not any(status.get('warning_counts',[]))
metrics={'success':bool(success),'strokes':report,'air_contact_count':int(a[air,25].sum()),'paper_force_peak_N':float(a[:,10].max()),'hand_paper_peak_N':float(a[:,22].max()),'shaft_paper_peak_N':float(a[:,23].max()),'hair_paper_max_overlap_mm':float(-a[:,24].min()*1000),'shaft_tilt_peak_deg':float(a[:,26].max()),'solver':status}
(P/(tag+'_writing_metrics.json')).write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
ax.axis('equal');ax.set(xlabel='Paper X (mm)',ylabel='Paper Y (mm)',title='Brush writing: target / actual physical contact');ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(P/(tag+'_overlay.png'),dpi=180)
