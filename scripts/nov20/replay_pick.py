
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os,sys,json,subprocess
from scipy.spatial.transform import Rotation as R
from env import episode,OUT,np
task=sys.argv[1];gpu='--gpu' in sys.argv;video='--video' in sys.argv
g=episode(task,video,gpu);meta=next(g);ix=meta['names'].index('item');pk=meta['names'].index('link_0_0')
src=sys.argv[2];f=sorted((OUT/(task+'_search')).glob('candidate*.npz'))[-1] if src=='last' else OUT/src
a0=np.load(f)['actions'];start=meta['poses'][ix,:3].copy()
rows=[];poses=[];actions=[];frames=[];ramp=lambda v:np.clip(v,0,1)
mount=R.from_euler('x',90,degrees=True);Rgoal=R.from_euler('zyx',[8,-29,0],degrees=True)
dr=R.identity();delta=np.zeros(3);zcmd=0
for i in range(720):
 a=a0[min(i,len(a0)-1)].copy();t=i/240
 if task=='scrape' and i>=389 and '--pick-only' not in sys.argv:
    if i==389:
     base=a.copy();op=poses[-1][ix,:3];ori=R.from_quat(poses[-1][ix,3:]);lever=op-poses[-1][pk,:3];startedge=op+ori.apply([-.025,0,-.003])
    a=base.copy();u=ramp((i-389)/60);qrot=R.from_rotvec(u*(Rgoal*ori.inv()).as_rotvec());qa=mount.inv()*qrot*mount*R.from_euler('XYZ',base[3:6]);a[3:6]=qa.as_euler('XYZ')
    edge_a=np.array([-.135,.000,.050]);contact_z=float(os.getenv('ZSET','.034'))
    dest=(1-u)*startedge+u*edge_a
    dest[2]+=(contact_z-edge_a[2])*ramp((i-449)/44)
    dest[0]+=.056*ramp((i-493)/92)-.056*ramp((i-614)/95)
    dest[1]+=.015*ramp((i-585)/29)
    cur=poses[-1][ix];edge=cur[:3]+R.from_quat(cur[3:]).apply([-.025,0,-.003]);dp=dest-(op+qrot.apply(lever)-lever+(qrot*ori).apply([-.025,0,-.003]))
    delta=np.clip(.99*delta+.025*(dest-edge),-.010,.010)
    if i>=493:zcmd=np.clip(zcmd+.000035*(rows[-1]['force_N']-.9),-.007,.004)
    a[:3]+=mount.inv().apply(dp+delta+[0,0,zcmd])
 z=g.send(a);p=z['poses'].copy();h=z['task_metrics'];h['lift_mm']=float((p[ix,2]-start[2])*1000);h['t']=t;rows.append(h);poses.append(p);actions.append(a)
 if z['frame'] is not None:frames.append(z['frame'])
tag=task+'_'+f.stem+('_gpu' if gpu else '_cpu')+os.getenv('VARIANT','')
out=OUT/'positive_review';out.mkdir(exist_ok=True)
(out/(tag+'.json')).write_text(json.dumps(rows))
np.savez_compressed(out/(tag+'.npz'),actions=actions,poses=poses,names=meta['names'])
if video:
 subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r','30','-i','-','-pix_fmt','yuv420p',str(out/(tag+'.mp4'))],input=np.asarray(frames).tobytes(),check=True)
print(tag,rows[-1],max(v['lift_mm'] for v in rows),flush=True)
