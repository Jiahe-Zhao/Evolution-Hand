
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from env import episode
from build import OUT
import sys, os,json,subprocess,numpy as np
task=sys.argv[1];negative='negative' in sys.argv;gpu='--gpu' in sys.argv;preview='--no-video' not in sys.argv
tag=os.getenv('TAG','')+task+('_negative' if negative else '')+('_gpu' if gpu else '')
control='--contrast' in sys.argv
if control:tag+='_contrast'
it=episode(task,preview,gpu);meta=next(it);names=meta['action_names'];frames=[];stats=[];actions=[];trajectory=[]
def ramp(v):return np.clip(v,0,1)
def target(a,name,value):a[names.index(name)]=value
for i in range(meta['horizon']):
 t=i*meta['dt'];a=meta['base'].copy()
 if not negative:
    if task=='carry':target(a,'diagnostic_wrist_x',.050*ramp((t-.4)/1.3))
    if task=='stone':
     for j in [2,3,4,5]:target(a,f'link_{j}_0_to_link_{j}_1',{2:1.10,3:1.0,4:.80,5:.70}[j]*ramp((t-.45)/.9))
    if task=='phone':
     target(a,'link_1_thumb_spread_joint',float(os.getenv('THUMB_SPREAD','.17'))*ramp((t-1.45)/.5))
     target(a,'link_1_0_to_link_1_1',.85*(1-ramp((t-.35)/.50)+ramp((t-float(os.getenv('RELEASE','.89')))/.40)-ramp((t-2.)/.5)))
    if task=='handle':
     target(a,'diagnostic_wrist_y',.119*(.5-.5*np.cos(np.pi*ramp((t-.45)/1.95))));target(a,'diagnostic_wrist_x',-.050*(.5-.5*np.cos(np.pi*ramp((t-.55)/1.8))))
     pass
 if control:
    if task=='carry':target(a,'diagnostic_wrist_x',.025*ramp((t-.4)/1.3))
    elif task=='stone':
     a=meta['base'].copy();target(a,'wrist_yaw',.65*ramp((t-.4)/.9))
    elif task=='handle':
     target(a,'diagnostic_wrist_y',min(.028,a[names.index('diagnostic_wrist_y')]))
     target(a,'diagnostic_wrist_x',0)
    elif task=='phone' and t>=1.4:
     target(a,'link_1_0_to_link_1_1',.85);target(a,'link_1_thumb_spread_joint',0)
 z=it.send(a);actions.append(a);stats.append(z['metrics'])
 if i%8==0:trajectory.append(z['poses'])
 if z['frame'] is not None:frames.append(z['frame'])
if preview:
 subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','480x360','-framerate','30','-i','-','-pix_fmt','yuv420p',str(OUT/(tag+'.mp4'))],input=np.asarray(frames).tobytes(),check=True)
np.savez_compressed(OUT/(tag+'.npz'),poses=trajectory,pose_stride=8,names=meta['names'],actions=actions,action_names=names,metrics=stats)
result=dict(task=task,negative=negative,contrast=control,backend='Newton / '+('MJWarp CUDA' if gpu else 'SolverMuJoCo CPU'),steps=i+1,dt=meta['dt'],success=z['success'],best_s=z['best_s'],events=z['events'],last=z['metrics'],columns='time drift goal_xy_error face_error_deg hinge_rad lift tilt_deg phone_phase hand_contact bail_contact'.split())
(OUT/(tag+'.json')).write_text(json.dumps(result,indent=2));print(result)
