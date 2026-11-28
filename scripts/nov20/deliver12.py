
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from grind12 import OUT,out,np,mj,subprocess
from score12 import assess
import sys,json,os,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
tag=sys.argv[1];f=np.load(out/(tag+'.npz'));x,ok,ts,dist,turn=assess(f);assert np.min(dist)>=30
hz=round(1/(x[1,0]-x[0,0]));period=x[ts[-1],0]-x[ts[0],0];beg,end=ts[0],ts[-1];win=slice(beg,end+1);valid=x[win][ok[win]];raw=x[win];bounds=[x[k,0] for k in ts];actual=[abs(x[k,1]-x[j,1])*1000 for j,k in zip(ts,ts[1:])]
fig,axs=plt.subplots(3,1,sharex=True,figsize=(11,7),gridspec_kw={'height_ratios':[2,1,1]})
t=x[win,0];axs[0].axhspan(.5,5,color='#d8efdd');axs[0].plot(t,raw[:,8],color='#38597a',lw=.55);axs[0].axhline(.5,color='#428453',lw=.8);axs[0].axhline(5,color='#428453',lw=.8);axs[0].set_ylabel('Normal load (N)');axs[0].set_title('Stone working face against the grinding bed | raw solver samples')
axs[1].plot(t,1000*(raw[:,1]-raw[0,1]),color='#805135',lw=1.1);axs[1].scatter(t[~ok[win]],(1000*(raw[:,1]-raw[0,1]))[~ok[win]],s=1,color='#ce5751',label='excluded sample');axs[1].legend(loc='upper right',fontsize=8);axs[1].set_ylabel('Stone travel (mm)')
axs[2].plot(t,raw[:,12],color='#2f7878',lw=.8);axs[2].set_ylabel('Stone tilt (deg)');axs[2].set_xlabel('Simulation time (s)')
for ax in axs:
 for vt in bounds[1:-1]:ax.axvline(vt,color='#74448b',ls='--',lw=.8)
 ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(out/'force_curve.png',dpi=160);plt.close(fig)
vm=mj.MjModel.from_xml_path(str(out/(tag+'.xml')));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];opt=mj.MjvOption();opt.geomgroup[5]=0
cams=[]
for target,d,az,el in [([-.060,.003,.038],.255,94,-36),([-.061,.003,.018],.200,71,-12)]:
 cam=mj.MjvCamera();cam.lookat[:]=target;cam.distance=d;cam.azimuth=az;cam.elevation=el;cams.append(cam)
names=f['names'];st=f['starts'];qs=f['q'];s=int(f['S']);cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-vcodec','libx264','-pix_fmt','yuv420p',str(out/'operation.mp4')];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
for k in range(0,len(qs),8):
 q=qs[k];idx=k*s
 for j,name in enumerate(names):
  id=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,str(name))
  if id<0:continue
  a=vm.jnt_qposadr[id];b=st[j]
  if vm.jnt_type[id]==0:vd.qpos[a:a+7]=np.r_[q[b:b+3],q[b+6],q[b+3:b+6]]
  else:vd.qpos[a]=q[b]
 mj.mj_forward(vm,vd);ren.update_scene(vd,cams[0]);left=ren.render().copy();vm.geom_group[hs]=5;ren.update_scene(vd,cams[1],scene_option=opt);right=ren.render().copy();vm.geom_group[hs]=0
 im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,23],fill=(24,34,38));leg=np.searchsorted(ts,idx,side='right');stage=f'pass {leg}/6' if 1<=leg<=6 else ('grasp and load' if leg==0 else 'complete');dr.text((8,5),f'hand_111 | {x[idx,0]:.2f}s | {stage}',fill='white');dr.text((488,5),f'Stone only | normal {x[idx,8]:.2f} N | tilt {x[idx,12]:.2f} deg',fill='white');proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close();assert proc.wait()==0;ren.close()
# Height difference across 70 mm long central working facet (yaw projected along trough).
from scipy.spatial.transform import Rotation as R
edge=[];bad=[];window_ratio=[]
for k in ts[1:-1]:
 w=x[k-round(.05*hz):k+round(.05*hz)+1];axis=R.from_quat(w[:,4:8]).as_matrix()[:,:,0]
 edge.append(float(np.abs(axis[:,2]).max()*70));bad.append(int(((w[:,9]<=.01)|(w[:,10]<=.01)).sum()));window_ratio.append(float(((w[:,8]>=.5)&(w[:,8]<=5)).mean()))
report=dict(edge_delta_mm=edge,one_end_samples=bad,turn_force_in_range_fraction=window_ratio,zero_load_s=float((raw[:,8]==0).sum()/hz),tag=tag,hz=hz,stroke_mm=dist.tolist(),actual_mm=actual,turn_windows=turn.tolist(),effective_N=[valid[:,8].min(),valid[:,8].max()],raw_N=[raw[:,8].min(),raw[:,8].max()],eligible_fraction=ok[win].mean(),turn_times=bounds,hold_fraction=raw[:,13].mean());(out/'metrics.json').write_text(json.dumps(report));print(report)
