from pathlib import Path
import csv,json,sys,collections,numpy as np
p=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_leaf04');tag=sys.argv[1];rows=list(csv.DictReader((p/(tag+'_trace.csv')).open()));trace={round(float(r['time_s']),5):r for r in rows};groups=collections.defaultdict(list);totals=collections.defaultdict(lambda:np.zeros(4))
for r in csv.DictReader((p/(tag+'_contacts.csv')).open()):
 t=round(float(r['time_s']),5);tr=trace[t]
 if tr['phase']!='hold_both':continue
 k=int(r['leaf']);theta=float(tr[f'leaf{k}_rad']);n=np.array([np.sin(theta),0,np.cos(theta)]);f=np.array([float(r[x]) for x in ['fx_N','fy_N','fz_N']]);proj=float(f@n);key=(k,r['type'],r['body_or_region']);groups[key].append((t,float(r['normal_N']),proj));totals[(k,t)]+=np.array([float(r['normal_N']),max(proj,0),max(-proj,0),proj])
result={}
for key,rr in groups.items():
 a=np.array(rr);result[str(key)]=dict(records=len(rr),max_point_normal_N=float(a[:,1].max()),mean_sample_normal_N=float(a[:,1].sum()/sum(r['phase']=='hold_both' for r in rows)),max_positive_plane_force_N=float(max(a[:,2].max(),0)),max_negative_plane_force_N=float(max(-a[:,2].min(),0)))
result['scope']='Sampled contacts; positive projection on upward leaf normal means force on hand toward leaf upper side. Not a substep impulse audit.'
(p/(tag+'_contact_analysis.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
