
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
o=OUT/'pen140';a,q=[np.load(o/'write.npz')[k] for k in ['a','q']];h=np.load(o/'hover.npz')['a'];start=np.loadtxt(o/'write_origin.txt');pos=start[:3];u=np.linspace(0,1,5001);curve=np.c_[pos[0]+.046*u,pos[1]+.006*np.sin(2*np.pi*u)];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];tree=cKDTree(curve)
def ink(x):return (x[:,9]>.01)&np.isfinite(x[:,7])
marked=ink(a);held=(a[:,10:12]>.01).all(axis=1);ok=marked&held;pts=a[ok,7:9];err,idx=tree.query(pts);bins=np.unique(np.minimum((arc[idx]/arc[-1]*100).astype(int),99));cov=len(bins);rms=np.sqrt(np.mean(err**2))*1000
ix=np.flatnonzero(ok);s=arc[idx];ints=sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ix[j+1]-ix[j]==1);end=-1;tot=0
for x,y in ints:tot+=max(0,y-max(x,end));end=max(end,y)
coverage=100*tot/arc[-1]
assert (h[:,9]==0).all() and (h[h[:,1]>0,10:12]>.01).all() and np.linalg.norm(h[-1,4:6]-h[-1,2:4])<.003 and not ink(h).any() and h[:,6].min()-(np.loadtxt(o/'hover_origin.txt')[3]+.001)>0 and coverage>=90 and rms<=3 and held[marked].all()
for label,data in [('write',a),('hover',h)]:np.savetxt(o/(label+'_contacts.csv'),data,delimiter=',',header='time_s,progress,target_x_m,target_y_m,tip_x_m,tip_y_m,tip_z_m,contact_x_m,contact_y_m,tip_paper_normal_N,thumb_pen_normal_N,index_pen_normal_N',comments='')
fig,ax=plt.subplots(figsize=(8,3.5));ax.plot(*(curve.T*1000),'--',color='darkorange',lw=2,label='Target');ax.plot(*(1000*np.where(ok[:,None],a[:,7:9],np.nan).T),color='navy',lw=1.3,label='Loaded tip contact');ax.set_aspect('equal');ax.set(xlabel='Paper X (mm)',ylabel='Paper Y (mm)',title=f'140 mm pen | covered {coverage:.2f}% | RMS {rms:.3f} mm');ax.legend();fig.tight_layout();fig.savefig(o/'overlay.png',dpi=150)
vm=mj.MjModel.from_xml_path(str(o/'write.xml'));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480,max_geom=12000);cam=mj.MjvCamera();cam.lookat[:]=[pos[0]+.066,pos[1]-.022,pos[2]+.040];cam.distance=.255;cam.azimuth=95;cam.elevation=-40;frames=[]
def uv(p):return (int((p[0]-pos[0]+.012)*6800)+490,int(175-(p[1]-pos[1])*6800))
paper=vm.geom('paper');z=vm.geom_pos[paper.id,2]+vm.geom_size[paper.id,2]
def stroke(p,b,z,r,color):
 g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32));mj.mjv_connector(g,mj.mjtGeom.mjGEOM_CAPSULE,r,np.r_[p,z],np.r_[b,z]);ren.scene.ngeom+=1
segments=[]
for run in np.split(np.flatnonzero(ok),np.flatnonzero(np.diff(np.flatnonzero(ok))!=1)+1):
 for i in range(0,len(run)-1,6):
  j=run[min(i+6,len(run)-1)];segments.append((j,a[run[i],7:9],a[j,7:9]))
def paper_marks(t):
 for i in range(0,5000,150):stroke(curve[i],curve[min(i+88,5000)],z+.00003,.00034,[.61,.33,.13,1])
 for j,p,b in segments:
  if j>t:break
  stroke(p,b,z+.00021,.00045,[.015,.16,.62,1])
for t in range(0,len(q),32):
 vd.qpos[:27]=q[t,:27];vd.qpos[27:30]=q[t,27:30];vd.qpos[30:]=q[t,[33,30,31,32]];mj.mj_forward(vm,vd);ren.update_scene(vd,cam);paper_marks(t);im=Image.new('RGB',(960,360),'#f3f1e9');im.paste(Image.fromarray(ren.render()),(0,0));dr=ImageDraw.Draw(im);dr.line([uv(p) for p in curve],fill='#bbb8af',width=4)
 for j in range(1,t+1):
  if marked[j-1] and marked[j]:dr.line([uv(a[j-1,7:9]),uv(a[j,7:9])],fill='#134573',width=3)
 dr.rectangle((0,0,960,23),fill='#172936');dr.text((8,5),'Paper: ochre guide / blue contact ink',fill='white');dr.text((489,5),f'Paper view | contact ink | t={a[t,0]:.2f}s | tip {a[t,9]:.3f} N',fill='white');frames.append(np.asarray(im))
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(o/'longpen_paper.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()
