"""Read-only acceptance audit for the AX50 task-13 active bail (提梁).

Inputs must already exist; this script never simulates and never reads IK targets:
  outputs/nov20/ax50_handle13/{tag}.npz        time, body_q, body_names, particles, tri, targets
  outputs/nov20/ax50_handle13/{tag}_trace.csv  time_s, phase, hinge_rad, lift_m, tilt_deg,
                                               body_contact_N, bail_contact_N, finite,
                                               substeps, qualified_substeps
Lift uses the model initial height before gravity. Pose metrics are independently
recomputed from body_q; contact qualification uses every-physics-step GPU reductions.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_handle13'
AXIS=np.array([0.,-1.,0.])
CRIT=dict(hinge_deg=45.,lift_m=.025,tilt_deg=15.,body_N=.0005,bail_N=.01,hold_s=.5)
TOL=dict(hinge_deg=.25,lift_m=.00025,tilt_deg=.25,time_s=1e-6)
COLS=['time_s','phase','hinge_rad','lift_m','tilt_deg','body_contact_N','bail_contact_N','finite','substeps','qualified_substeps']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def short_names(a):return [x.decode() if isinstance(x,bytes) else str(x) for x in a]
def quat_to_matrix(q):
    q=np.asarray(q,float);q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    x,y,z,w=q[...,0],q[...,1],q[...,2],q[...,3];m=np.empty(q.shape[:-1]+(3,3))
    m[...,0,0]=1-2*(y*y+z*z);m[...,0,1]=2*(x*y-w*z);m[...,0,2]=2*(x*z+w*y)
    m[...,1,0]=2*(x*y+w*z);m[...,1,1]=1-2*(x*x+z*z);m[...,1,2]=2*(y*z-w*x)
    m[...,2,0]=2*(x*z-w*y);m[...,2,1]=2*(y*z+w*x);m[...,2,2]=1-2*(x*x+y*y)
    return m
def _skew(R):return np.stack([R[...,2,1]-R[...,1,2],R[...,0,2]-R[...,2,0],R[...,1,0]-R[...,0,1]],-1)
def _signed_about(R,axis):return np.arctan2(.5*(_skew(R)@axis),(np.trace(R,axis1=-2,axis2=-1)-1)/2)
def _wrap(d):return (np.asarray(d)+np.pi)%(2*np.pi)-np.pi
def _bool(v):return str(v).strip().lower() in ('true','1')
def _longest(ok,t):
    best=None;start=None
    for i,v in enumerate(list(ok)+[False]):
        if v and start is None:start=i
        elif not v and start is not None:
            if best is None or t[i-1]-t[start]>t[best[1]]-t[best[0]]:best=(start,i-1)
            start=None
    return best

def _audit(tag):
    npz=OUT/(tag+'.npz');tp=OUT/(tag+'_trace.csv')
    for p in (npz,tp):
        if not p.exists():raise FileNotFoundError(f'missing recorded input: {p}')
    z=np.load(npz);t=np.asarray(z['time'],float).ravel();q=np.asarray(z['body_q'],float)
    if q.ndim!=3 or q.shape[0]!=len(t) or q.shape[2]!=7:raise ValueError(f'body_q shape {q.shape} does not match {len(t)} frames of pos+quat')
    names=short_names(z['body_names']);sh=[n.rsplit('/',1)[-1] for n in names]
    if sh[-2:]==['item','bail']:ii,bi=len(sh)-2,len(sh)-1
    elif 'item' in sh and 'bail' in sh:ii,bi=sh.index('item'),sh.index('bail')
    else:raise ValueError(f'item/bail not found in body_names {sh}')
    with tp.open() as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'empty trace {tp}')
    miss=[c for c in COLS if c not in rows[0]]
    if miss:raise ValueError(f'trace missing columns {miss}')
    tr={k:np.array([float(r[k]) for r in rows]) for k in COLS if k not in ('phase','finite')}
    finite=np.array([_bool(r['finite']) for r in rows]);phase=[r['phase'] for r in rows]
    p=q[:,:,:3];R=quat_to_matrix(q[:,:,3:])
    hinge=np.unwrap(_signed_about(np.einsum('tji,tjk->tik',R[:,ii],R[:,bi]),AXIS))
    hinge_deg=np.degrees(hinge)
    meta=json.loads((OUT/(tag+'_model.json')).read_text())
    lift=p[:,ii,2]-meta['object_initial'][0][2]
    tilt=np.degrees(np.arccos(np.clip(R[:,ii,2,2],-1,1)))
    n=min(len(t),len(rows));frames_npz,frames_trace=len(t),len(rows);t0,tend=float(t[0]),float(t[-1])
    lift0=float(tr['lift_m'][0])
    t=t[:n];hinge=hinge[:n];hinge_deg=hinge_deg[:n];lift=lift[:n];tilt=tilt[:n]
    body=tr['body_contact_N'][:n];bail=tr['bail_contact_N'][:n];sub=tr['substeps'][:n];qual=tr['qualified_substeps'][:n]
    finite=finite[:n];phase=phase[:n]
    dt=float(np.median(np.diff(t))) if len(t)>1 else 0.
    integrity=dict(frames_npz=frames_npz,frames_trace=frames_trace,frames_match=bool(frames_npz==frames_trace),
        time_max_error_s=float(np.max(np.abs(tr['time_s'][:n]-t[:n]))),
        hinge_max_error_deg=float(np.max(np.abs(np.degrees(_wrap(hinge[:n]-tr['hinge_rad'][:n]))))),
        lift_max_error_m=float(np.max(np.abs(lift[:n]-tr['lift_m'][:n]))),
        lift_origin_offset_m=0.,
        tilt_max_error_deg=float(np.max(np.abs(tilt[:n]-tr['tilt_deg'][:n]))),
        npz_poses_finite=bool(np.isfinite(q).all()),
        increasing=bool(np.all(np.diff(t)>0)),
        last_two_bodies_are_item_bail=bool(sh[-2:]==['item','bail']))
    integrity['ok']=bool(integrity['frames_match'] and integrity['time_max_error_s']<=TOL['time_s'] and integrity['hinge_max_error_deg']<=TOL['hinge_deg'] and integrity['lift_max_error_m']<=TOL['lift_m'] and integrity['tilt_max_error_deg']<=TOL['tilt_deg'] and integrity['npz_poses_finite'] and integrity['increasing'] and integrity['last_two_bodies_are_item_bail'])
    qualified=np.array([int(a)==int(b) and int(b)>0 for a,b in zip(qual,sub)])
    ok=((hinge_deg>=CRIT['hinge_deg'])&(lift>=CRIT['lift_m'])&(tilt<=CRIT['tilt_deg'])&(body<=CRIT['body_N'])&(bail>CRIT['bail_N'])&qualified&finite)
    ok=ok[:n]
    run=_longest(ok,t[:n])
    hold={}
    if run:
        a,b=run;gaps=np.diff(t[a:b+1]);contig=bool(len(gaps)==0 or np.max(gaps)<=1.5*dt+1e-12)
        hold=dict(interval_s=[float(t[a]),float(t[b])],frames=int(b-a+1),duration_s=float(t[b]-t[a]),
            contiguous=contig,ok=bool(t[b]-t[a]>=CRIT['hold_s']-1e-12 and contig),
            min_hinge_deg=float(np.min(hinge_deg[a:b+1])),min_lift_m=float(np.min(lift[a:b+1])),
            max_tilt_deg=float(np.max(tilt[a:b+1])),max_body_N=float(np.max(body[a:b+1])),
            min_bail_N=float(np.min(bail[a:b+1])),min_qualified_ratio=float(np.min(qual[a:b+1]/np.maximum(sub[a:b+1],1))))
        hold['all_substeps']=dict(min_hinge_deg=min(float(x['min_hinge_deg']) for x in rows[a:b+1]),min_lift_m=min(float(x['min_lift_m']) for x in rows[a:b+1]),max_tilt_deg=max(float(x['max_tilt_deg']) for x in rows[a:b+1]),min_bail_N=min(float(x['min_bail_contact_N']) for x in rows[a:b+1]),max_body_N=max(float(x['max_body_contact_N']) for x in rows[a:b+1]))
    else:hold=dict(interval_s=None,frames=0,duration_s=0.,contiguous=False,ok=False)
    anywhere=max(float(r['max_body_contact_N']) for r in rows)
    crit=dict(
        bail_hinge=dict(max_deg=float(np.max(hinge_deg)),required_deg=CRIT['hinge_deg'],ok=bool(np.max(hinge_deg)>=CRIT['hinge_deg'])),
        item_lift=dict(max_m=float(np.max(lift)),required_m=CRIT['lift_m'],ok=bool(np.max(lift)>=CRIT['lift_m'])),
        item_tilt=dict(max_deg=float(np.max(tilt)),required_max_deg=CRIT['tilt_deg'],ok=bool(np.max(tilt)<=CRIT['tilt_deg'])),
        hand_item_contact=dict(max_N=anywhere,required_max_N=CRIT['body_N'],ok=bool(anywhere<=CRIT['body_N'])),
        hand_bail_load=dict(max_N=float(np.max(bail)),required_N=CRIT['bail_N'],ok=bool(np.max(bail)>CRIT['bail_N'])),
        substep_qualification=dict(frames_failed=int(np.sum(~qualified)),min_ratio=float(np.min(qual/np.maximum(sub,1))),ok=bool(qualified.all())),
        finite=dict(frames_failed=int(np.sum(~finite)),ok=bool(finite.all())),
        duration=dict(span_s=tend-t0,ok=bool(tend-t0>=CRIT['hold_s'])))
    status=None;sp=OUT/(tag+'_status.json')
    if sp.exists():
        try:status=json.loads(sp.read_text())
        except ValueError:status=dict(stop='unreadable_status_file')
    checks=dict(integrity=integrity['ok'],no_hand_item_contact=crit['hand_item_contact']['ok'] and not any(int(r['ever_body_contact']) for r in rows),
        hold=hold['ok'],all_finite=crit['finite']['ok'],duration=crit['duration']['ok'],
        status_completed=bool(status is not None and status.get('stop')=='completed'))
    steady_run=_longest(ok & np.array([x=='hold' for x in phase]),t)
    steady=dict(interval_s=None,duration_s=0.,ok=False)
    if steady_run:
        sa,sb=steady_run;duration=float(t[sb]-t[sa]);steady=dict(interval_s=[float(t[sa]),float(t[sb])],duration_s=duration,ok=bool(duration>=CRIT['hold_s']-1e-12))
    checks['stationary_hold']=steady['ok']
    result=dict(tag=tag,passed=bool(all(checks.values())),checks=checks,stationary_hold=steady,
        scope='recorded files only; criteria recomputed from body_q with model initial height; no simulation, no IK targets, no import check counted',
        criteria=crit,hold=hold,integrity=integrity,phases=sorted(set(phase)),
        initial=dict(time_s=t0,item_pos=p[0,ii].tolist(),item_quat=q[0,ii,3:].tolist(),
            bail_quat=q[0,bi,3:].tolist(),source='model object_initial height; recorded body_q; IK targets not used'),
        final=dict(time_s=tend,item_pos=p[-1,ii].tolist(),bail_hinge_deg=float(hinge_deg[-1]),lift_m=float(lift[-1]),tilt_deg=float(tilt[-1])),
        contact=dict(hand_item_max_N_anywhere=anywhere,hand_bail_max_N_anywhere=float(np.max(bail)),
            hand_item_max_N_inside_hold=hold.get('max_body_N'),hand_item_contact_outside_hold=bool(any(body[i]>CRIT['body_N'] for i in range(n) if not (run and run[0]<=i<=run[1])))),
        simulation_status=status,files=dict(npz=str(npz),npz_sha256=sha(npz),trace=str(tp),trace_sha256=sha(tp)),
        targets_present='targets' in z.files,median_dt_s=dt)
    series=dict(time=t[:n],hinge_deg=hinge_deg[:n],lift_m=lift[:n],tilt_deg=tilt[:n],body_N=body[:n],bail_N=bail[:n],
        substeps=sub[:n],qualified_substeps=qual[:n],ok=ok,phase=phase[:n])
    return result,series

def audit(tag,series=False):
    r,s=_audit(tag)
    return (r,s) if series else r

if __name__=='__main__':
    import sys
    tag=sys.argv[1] if len(sys.argv)>1 else 'handle_v1'
    r=audit(tag)
    (OUT/(tag+'_result.json')).write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['passed'] else 1)
