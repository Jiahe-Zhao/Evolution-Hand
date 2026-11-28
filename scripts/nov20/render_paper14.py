
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import csv,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=OUT/'pen14';a,q=[np.load(o/'write.npz')[k] for k in ['a','q']];h=np.load(o/'hover.npz')['a'];pos=np.load(o/'origin.npy');u=np.linspace(0,1,4001);curve=np.c_[pos[0]-.002+.046*u,pos[1]+.006*np.sin(2*np.pi*u)];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];tree=cKDTree(curve)
def ink(x):return (x[:,9]>.01)&np.isfinite(x[:,7])
marked=ink(a);held=(a[:,10:12]>.01).all(axis=1);ok=marked&held;pts=a[ok,7:9];err,idx=tree.query(pts);bins=np.unique(np.minimum((arc[idx]/arc[-1]*100).astype(int),99));cov=len(bins);rms=np.sqrt(np.mean(err**2))*1000
ix=np.flatnonzero(ok);s=arc[idx];ints=sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ix[j+1]-ix[j]==1);end=-1;tot=0
for x,y in ints:tot+=max(0,y-max(x,end));end=max(end,y)
coverage=100*tot/arc[-1]
assert (h[:,9]==0).all() and (h[h[:,1]>0,10:12]>.01).all() and np.linalg.norm(h[-1,4:6]-curve[-1])<.003 and not ink(h).any() and h[:,6].min()-(pos[2]-.0434)>0 and coverage>=90 and rms<=3 and held[marked].all()
vm=mj.MjModel.from_xml_path(str(o/'write.xml'));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480,max_geom=12000);cam=mj.MjvCamera();cam.lookat[:]=[pos[0]+.014,pos[1],pos[2]-.030];cam.distance=.128;cam.azimuth=95;cam.elevation=-58;frames=[]
def uv(p):return (int((p[0]-pos[0]+.012)*6800)+490,int(175-(p[1]-pos[1])*6800))
paper=vm.geom('paper');z=vm.geom_pos[paper.id,2]+vm.geom_size[paper.id,2]
def stroke(p,b,z,r,color):
 g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32));mj.mjv_connector(g,mj.mjtGeom.mjGEOM_CAPSULE,r,np.r_[p,z],np.r_[b,z]);ren.scene.ngeom+=1
segments=[]
for run in np.split(np.flatnonzero(ok),np.flatnonzero(np.diff(np.flatnonzero(ok))!=1)+1):
 for i in range(0,len(run)-1,6):
  j=run[min(i+6,len(run)-1)];segments.append((j,a[run[i],7:9],a[j,7:9]))
def paper_marks(t):
 for i in range(0,4000,120):stroke(curve[i],curve[min(i+70,4000)],z+.00003,.00034,[.61,.33,.13,1])
 for j,p,b in segments:
  if j>t:break
  stroke(p,b,z+.00021,.00045,[.015,.16,.62,1])
for t in range(0,len(q),32):
 vd.qpos[:27]=q[t,:27];vd.qpos[27:30]=q[t,27:30];vd.qpos[30:]=q[t,[33,30,31,32]];mj.mj_forward(vm,vd);ren.update_scene(vd,cam);paper_marks(t);im=Image.new('RGB',(960,360),'#f3f1e9');im.paste(Image.fromarray(ren.render()),(0,0));dr=ImageDraw.Draw(im);dr.line([uv(p) for p in curve],fill='#bbb8af',width=4)
 for j in range(1,t+1):
  if marked[j-1] and marked[j]:dr.line([uv(a[j-1,7:9]),uv(a[j,7:9])],fill='#134573',width=3)
 dr.rectangle((0,0,960,23),fill='#172936');dr.text((8,5),'Paper: ochre guide / blue contact ink',fill='white');dr.text((489,5),f'Paper view | contact ink | t={a[t,0]:.2f}s | tip {a[t,9]:.3f} N',fill='white');frames.append(np.asarray(im))
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(o/'operation_paper.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()
