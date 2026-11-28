
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from visibility04 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import subprocess
z=np.load(o/'hand.npz');a=z['a'];q=z['q'];t=a[:,0];v=np.loadtxt(o/'hand_visibility.csv',delimiter=',',skiprows=1)[:,1];i=np.flatnonzero((v>=.8)&(a[:,3]>.01)&(a[:,7]>.01)&(a[:,[4,5,6,8]].sum(1)==0));w=max(np.split(i,np.where(np.diff(i)>1)[0]+1),key=len);l,h=w[[0,-1]];dur=t[h]-t[l];assert dur>=.5 and a[433:1220,3].min()>.01 and v[480]<.8 and max(abs(a[-1,1:3]))<1
p=[]
for k in range(2):
 u=np.load(o/(str(k)+'.npz'))['a'];p.append((u[:,1+k].max(),abs(u[:,2-k]).max(),max(abs(u[-1,1:3]))));assert p[-1][0]>45 and max(p[-1][1:])<1
msg=f"""## 04 拨叶翻找：固定初态通过
旧版两片自由薄盒分别受推后没有回到遮挡位；简化对照中推动下叶还会带动上叶。本次保留原场景，另建两片各自带被动叶柄的刚性叶片，重力、扭簧（0.0014 N·m/rad）及阻尼提供回落。单叶依次单独受力，峰值 {p[0][0]:.2f}° / {p[1][0]:.2f}°，另一片最大偏角 {p[0][1]:.2f}° / {p[1][1]:.2f}°；撤力后两组终态偏角均小于 1°。

Newton SolverMuJoCo CPU，480 Hz，hand_111 左手。食指末节拨开并保持第一片，中指末节再拨开第二片；正例仅驱动手。固定俯视相机对目标顶面均匀采样 2240 条射线，叶片、叶柄和手都参与遮挡。初始可见 {100*v[0]:.1f}%，只开第一片（1.00 s）为 {100*v[480]:.1f}%；从 {t[l]:.3f} 至 {t[h]:.3f} s 连续 {dur:.3f} s 可见率 ≥80%，同时两指定末节各自法向载荷 >0.01 N，无其他手部对叶的有载接触（单点记录门槛 0.0005 N）。2.00 s 可见 {100*v[960]:.1f}%，叶角 {a[960,1]:.1f}° / {a[960,2]:.1f}°。双指撤回后末帧可见 {100*v[-1]:.1f}%。此项验证刚性叶片绕叶柄的独立运动，不验证叶面弯曲或随机初态成功率。

证据：outputs/nov20/leaf04/operation.mp4、visibility.png、hand_contacts.csv（leaf 0/1 为第一/第二片，link_2_3 / link_3_3 为食指/中指末节）、hand_visibility.csv、hand.npz、单叶对照 0.npz / 1.npz。复现：scripts/nov20/leaf04.py 分别运行 0、1、hand，再运行 visibility04.py hand 和 report04.py。
"""
(o/'RESULTS.md').write_text(msg);print(msg,flush=True)
fig,ax=plt.subplots(3,1,figsize=(9,6),sharex=True,layout='constrained')
ax[0].plot(t,v*100,color='#267348');ax[0].axhline(80,ls='--',color='gray');ax[0].set_ylabel('Visible area (%)')
for j,label in enumerate(('1 / index','2 / middle')):
 ax[1].plot(t,a[:,j+1],label=label);ax[2].plot(t,a[:,3+j*4],label=label)
ax[1].legend();ax[1].set_ylabel('Leaf angle (deg)');ax[2].set_ylabel('Tip normal load (N)');ax[2].set_xlabel('Simulation time (s)')
for b in ax:b.axvspan(t[l],t[h],color='#56a37a',alpha=.13);b.grid(alpha=.2)
fig.savefig(o/'visibility.png',dpi=150)
m=mj.MjModel.from_xml_path(str(o/'hand.xml'));d=mj.MjData(m);m.vis.global_.offwidth=640;m.vis.global_.offheight=420;ren=mj.Renderer(m,420,640);cam=mj.MjvCamera();cam.lookat[:]=[-.089,.009,.081];cam.distance=.23;cam.azimuth=115;cam.elevation=-26
font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),22)
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x500','-framerate','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/'operation.mp4')],stdin=subprocess.PIPE)
for k in range(0,len(t),8):
 d.qpos[:]=q[k];mj.mj_forward(m,d);im=Image.new('RGB',(1280,500),'#222a30')
 for j,camera in enumerate([cam,'auditTop']):
  ren.update_scene(d,camera=camera);im.paste(Image.fromarray(ren.render()),(640*j,40))
 dr=ImageDraw.Draw(im);dr.text((15,6),'Hand / leaf contact',font=font);dr.text((655,6),'Fixed audit view  |  orange = target',font=font);dr.text((15,466),f'{t[k]:.2f} s   |   0.5x speed       Index {a[k,3]:.3f} N      Middle {a[k,7]:.3f} N',font=font);dr.text((810,466),f'Target visible: {v[k]*100:.1f} %',font=font);proc.stdin.write(im.tobytes())
 if k==960:im.resize((768,300)).save(o/'preview.jpg')
proc.stdin.close();assert proc.wait()==0;ren.close()
