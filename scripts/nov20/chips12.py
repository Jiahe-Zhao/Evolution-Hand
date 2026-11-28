
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
vm=mj.MjModel.from_xml_path(str(out/(tag+'.xml')));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];opt=mj.MjvOption();opt.geomgroup[5]=0
cams=[]
for target,d,az,el in [([-.060,.003,.038],.255,94,-36),([-.061,.003,.018],.200,71,-12)]:
 cam=mj.MjvCamera();cam.lookat[:]=target;cam.distance=d;cam.azimuth=az;cam.elevation=el;cams.append(cam)
names=f['names'];st=f['starts'];qs=f['q'];s=int(f['S']);cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-vcodec','libx264','-pix_fmt','yuv420p',str(out/'wood_fx.mp4')];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
rng=np.random.default_rng(19);chips=[];credit=0.;prev_idx=0
def draw_chips(tm):
 for born,xyz,velocity,size,col,landing in chips:
  dt=tm-born
  if dt>1.3:continue
  fall=min(dt,landing);pos=xyz+fall*velocity+[0,0,-.5*9.81*fall**2];sc=ren.scene
  if sc.ngeom>=sc.maxgeom:break
  mj.mjv_initGeom(sc.geoms[sc.ngeom],mj.mjtGeom.mjGEOM_ELLIPSOID,size,pos,np.eye(3).flatten(),col);sc.ngeom+=1
for k in range(0,len(qs),8):
 q=qs[k];idx=k*s
 for j,name in enumerate(names):
  id=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,str(name))
  if id<0:continue
  a=vm.jnt_qposadr[id];b=st[j]
  if vm.jnt_type[id]==0:vd.qpos[a:a+7]=np.r_[q[b:b+3],q[b+6],q[b+3:b+6]]
  else:vd.qpos[a]=q[b]
 work=np.maximum(-x[prev_idx:idx,18],0)*ok[prev_idx:idx];credit+=np.sum(work)/hz if beg<=idx<=end else 0.;prev_idx=idx
 while credit>.00046:
  credit-=.00046;side=rng.choice([-1,1]);xx=x[idx,1]+rng.uniform(-.026,.026);yy=x[idx,2]+side*.021
  bedz=.011+np.interp(abs(yy-.003),[0,.010,.027,.045],[0,0,.0012,.0035]);z=bedz+.0009;vz=rng.uniform(.21,.46);land=(vz+np.sqrt(vz*vz+2*9.81*.0006))/9.81
  chips.append((x[idx,0],np.array([xx,yy,z]),np.array([rng.uniform(-.03,.03),side*rng.uniform(.03,.10),vz]),np.array([rng.uniform(.0006,.0016),.00042,.00030]),np.array([.82,rng.uniform(.53,.70),.26,1],np.float32),land))
 mj.mj_forward(vm,vd);ren.update_scene(vd,cams[0]);draw_chips(x[idx,0]);left=ren.render().copy();vm.geom_group[hs]=5;ren.update_scene(vd,cams[1],scene_option=opt);draw_chips(x[idx,0]);right=ren.render().copy();vm.geom_group[hs]=0
 im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,23],fill=(24,34,38));leg=np.searchsorted(ts,idx,side='right');stage=f'pass {leg}/6' if 1<=leg<=6 else ('grasp and load' if leg==0 else 'complete');dr.text((8,5),f'hand_111 | {x[idx,0]:.2f}s | {stage}',fill='white');dr.text((488,5),f'Visual chips only | normal {x[idx,8]:.2f} N | tilt {x[idx,12]:.2f} deg',fill='white');proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close();assert proc.wait()==0;ren.close()

assert np.array_equal(x[:,:15],np.load(out/'verified.npz')['data']);vel=np.gradient(raw[:,1])*hz;mask=ok[win]&(abs(vel)>.005);result=dict(mu=.34,median_tangent_N=float(np.median(np.linalg.norm(raw[mask,15:18],axis=1))),opposing_x_fraction=float(np.mean(raw[mask,15]*vel[mask]<0)),dissipated_J=float(-raw[:,18].sum()/hz),visual_chips=int(len(chips)));(out/'friction.json').write_text(json.dumps(result));print(result)
