
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import subprocess
o=OUT/'fruit02';A=np.array([-.093,.003,.127]);m=mj.MjModel.from_xml_path(str(o/'pick.xml'));d=mj.MjData(m);m.vis.global_.offwidth=760;m.vis.global_.offheight=480;ren=mj.Renderer(m,480,760);cam=mj.MjvCamera();cam.lookat[:]=[-.086,.004,.083];cam.distance=.21;cam.azimuth=45;cam.elevation=-12
fig,ax=plt.subplots(2,1,figsize=(8,5),sharex=True,layout='constrained');stats={};font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),21)
for tag in ['tap','pick']:
 z=np.load(o/(tag+'.npz'));a=z['a'];q=z['q'];t=a[:,0];ax[0].plot(t,a[:,1]*a[:,2],label=tag);idx=np.flatnonzero(a[:,3]);
 if tag=='tap':
  assert not len(idx) and a[:,4:6].max()>.01;stats['tap']=(a[:,1].max(),a[:,4:6].max())
 else:
  ib=idx.item();sel=t>=t[-1]-.51;slip=np.linalg.norm(a[sel,8:11]-a[sel,8:11][0],axis=1).max()*1000
  assert np.ptp(t[sel])>=.5 and t[sel][0]>t[ib] and a[sel,4:6].min()>.01 and a[sel,7].max()==0 and slip<2 and np.all(a[ib,4:6]>.01)
  stats['pick']=(t[ib],a[ib,1],np.ptp(t[sel]),slip,*a[sel,4:6].min(axis=0));ax[1].plot(t,a[:,4],label='thumb tip');ax[1].plot(t,a[:,5],label='index tip');ax[0].scatter(t[ib],a[ib,1],s=30,label='break demand');ax[0].axvline(t[ib],ls=':',color='gray');ax[1].axvspan(t[sel][0],t[-1],color='#95cfa5',alpha=.3)
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','760x560','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/(tag+'.mp4'))],stdin=subprocess.PIPE)
 for k in range(0,len(t),8):
  d.qpos[:27]=q[k,:27];d.qpos[27:30]=q[k,27:30];d.qpos[30:34]=q[k,[33,30,31,32]];mj.mj_forward(m,d);ren.update_scene(d,camera=cam);sc=ren.scene;g=sc.geoms[sc.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.43,.54,.15,1],np.float32));end=q[k,-7:-4]+R.from_quat(q[k,-4:]).apply([0,0,.014]) if a[k,2] else A+[0,0,-.028];mj.mjv_connector(g,mj.mjtGeom.mjGEOM_CAPSULE,.0013,A,end);sc.ngeom+=1
  im=Image.new('RGB',(760,560),'#202934');im.paste(Image.fromarray(ren.render()),(0,40));dr=ImageDraw.Draw(im);dr.text((12,6),f'02 | {tag}     {t[k]:.2f} s    '+('ATTACHED' if a[k,2] else 'DETACHED'),font=font);dr.text((12,524),f'Stem: {a[k,1]:.3f} N      Thumb: {a[k,4]:.2f} N      Index: {a[k,5]:.2f} N',font=font);proc.stdin.write(im.tobytes())
  if tag=='pick' and k==960:im.resize((570,420)).save(o/'preview.jpg')
 proc.stdin.close();assert proc.wait()==0
ren.close();ax[0].axhline(.45,ls='--',label='break threshold');ax[0].set_ylabel('Applied stem tension (N)');ax[1].set_ylabel('Normal contact (N)');ax[1].set_xlabel('Simulation time (s)')
for b in ax:b.legend();b.grid(alpha=.2)
fig.savefig(o/'force.png',dpi=160)
msg="""## 02 抓取果实：枝上采摘固定初态验收
12 g、约 30 × 24 × 28 mm 自由果实（两侧略扁的凸网格），由顶部连接到固定枝条。果柄是仅承拉的等效弹簧阻尼元件，自然长 30 mm、刚度 55 N/m、轴向阻尼 0.18 N·s/m；张力需求首次达到 0.45 N 即不可逆断开。断开步记录触发需求值，实际施加的果柄力从该步归零。不模拟木质纤维撕裂，阈值非实测品种参数。

hand_111 左手，Newton SolverMuJoCo CPU，480 Hz。轻碰对照有真实指尖接触，单指尖峰值 %.3f N，全程果柄需求峰值 %.3f N，未脱离。正例先由拇指与食指末节夹住，再移腕下拉；%.3f s 时有双指接触并触发断柄（%.3f N）。末段连续 %.3f s 两指同时承载，无地面或枝条支撑；果心相对掌坐标距该段起点最大漂移 %.3f mm，双指最小法向载荷 %.3f / %.3f N。果实未绑定到手，无脱离后的辅助力。本记录对应枝上采摘，不代表旧版台面三指抓取或随机初态通过。

证据：outputs/nov20/fruit02/ 的 pick.mp4、tap.mp4（半速）、force.png、pick_contacts.csv、tap_contacts.csv、pick_trace.csv、tap_trace.csv、XML/NPZ；脚本 scripts/nov20/fruit02.py 与 report02.py。
"""%(stats['tap'][1],stats['tap'][0],*stats['pick']);(o/'RESULTS.md').write_text(msg);print(msg)
