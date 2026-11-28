
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from audit_handle import *
import sys,collections
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
tag=sys.argv[1];r,s=audit(tag,series=True);rows=list(csv.DictReader((OUT/(tag+'_contacts.csv')).open()));tt=s['time'];lookup={round(t,6):i for i,t in enumerate(tt)};per={};xyz=np.zeros((len(tt),3));norm=np.zeros((len(tt),2))
for row in rows:
 i=lookup[round(float(row['time_s']),6)];part=row['type']+':'+row['hand_part'];v=per.setdefault(part,np.zeros(len(tt)));f=float(row['normal_N']);v[i]+=f;norm[i,0 if row['object']=='item' else 1]+=f;xyz[i]+=[float(row[k]) for k in ['fx_N','fy_N','fz_N']]
a,b=r.get('stationary_hold',r['hold'])['interval_s'] or [1e10,1e10];mask=(tt>=a)&(tt<=b)
summary=dict(hold_interval_s=[a,b] if mask.any() else None,endpoint_normal_sum_max_error_N=float(np.max(np.abs(norm-np.stack([s['body_N'],s['bail_N']],1)))),parts={k:dict(max_normal_N=float(v.max()),hold_contact_fraction=float(np.mean(v[mask]>.001)) if mask.any() else None,hold_mean_normal_N=float(np.mean(v[mask])) if mask.any() else None) for k,v in per.items()},hold_mean_force_on_hand_N=xyz[mask].mean(0).tolist() if mask.any() else None,scope='100 Hz individual contact record; every physics substep qualification separately in trace')
(OUT/(tag+'_contact_summary.json')).write_text(json.dumps(summary,indent=2));fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
for k,v in per.items():ax[0].plot(tt,v,label=k)
ax[0].legend();ax[0].set_ylabel('summed normal force (N)')
for i,k in enumerate(['Fx','Fy','Fz']):ax[1].plot(tt,xyz[:,i],label=k)
ax[1].legend();ax[1].set_ylabel('force on hand (N)');ax[1].set_xlabel('time (s)')
for x in ax:
 x.grid(alpha=.3)
 if mask.any():x.axvspan(a,b,color='g',alpha=.12)
fig.tight_layout();fig.savefig(OUT/(tag+'_contact_curves.png'),dpi=160);print(json.dumps(summary,indent=2))
