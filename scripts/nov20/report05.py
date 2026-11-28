
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import csv,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
o=OUT/'strike05';fig,ax=plt.subplots(2,1,figsize=(8.2,5.6),layout='constrained');font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),21);records={}
for tag in ['bare','hit']:
 z=np.load(o/(tag+'.npz'));q=z['q'];a=z['a'];t=a[:,0];F=a[:,1];H=a[:,2];grip=((a[:,5:10]>.01).sum(1)>=2);held=grip&(a[:,3]<.001);duration=0;passed=False
 armed=False
 for j in range(len(t)):
  if duration>=2:armed=True
  if not held[j]:armed=False
  if F[j]>=5 and H[j]<.001 and grip[j] and armed:passed=True
  duration=duration+1/960 if held[j] and F[j]<.001 else 0
 if tag=='bare':assert not passed and H.max()>5 and F.max()==0;bare=float(H.max())
 else:
  assert passed and H.max()<.001
  j=np.flatnonzero(F>.001)[0];i=j-1
  while i>=0 and held[i] and F[i]<.001:i-=1
  hold=(j-1-i)/960;tf=t[j];jp=np.argmax(F);peak=F[jp];tp=t[jp];speed=(a[j-1,-1]-a[j-5,-1])*240
  parts=list(csv.DictReader(open(o/'hit_contacts.csv')));links=sorted({p['link'] for p in parts if abs(float(p['time_s'])-tp)<.0004 and p['object_geom']!='nut'});assert hold>=2
  ax[0].plot(t,a[:,4:10].sum(1),label='hand on stone (normal sum)',color='#8d6946',lw=1);ax[0].axvspan(t[i+1],tf,color='green',alpha=.1)
 ax[1].plot(t,F if tag=='hit' else H,label=('stone on target' if tag=='hit' else 'bare hand on target (control)'))
 m=mj.MjModel.from_xml_path(str(o/(tag+'.xml')));d=mj.MjData(m);m.vis.global_.offwidth=800;m.vis.global_.offheight=500;ren=mj.Renderer(m,500,800);cam=mj.MjvCamera();cam.lookat[:]=[-.027,-.004,.07];cam.azimuth=-45;cam.elevation=-12;cam.distance=.23
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','800x584','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/(tag+'.mp4'))],stdin=subprocess.PIPE)
 for frame,k in enumerate(list(range(0,len(t),32))+list(range(int(2.68*960),int(2.94*960),2))):
  d.qpos[:27]=q[k,:27]
  if tag=='hit':d.qpos[27:30]=q[k,27:30];d.qpos[30:34]=q[k,[33,30,31,32]]
  mj.mj_forward(m,d);ren.update_scene(d,camera=cam);im=Image.new('RGB',(800,584),'#222a30');im.paste(Image.fromarray(ren.render()),(0,42));dr=ImageDraw.Draw(im);dr.text((12,8),'05 | '+('Hammerstone' if tag=='hit' else 'Bare hand control')+(' | 1/16x replay' if frame>=len(range(0,len(t),32)) else '')+f'    t = {t[k]:.3f} s',font=font);dr.text((12,550),f'Stone-target: {F[k]:.2f} N     Hand-target: {H[k]:.2f} N',font=font);proc.stdin.write(im.tobytes())
  if tag=='hit' and abs(k-jp)<2:im.resize((560,409)).save(o/'preview.jpg')
 proc.stdin.close();assert proc.wait()==0;ren.close()
ax[0].legend();ax[0].set(ylabel='Grasp load (N)',xlabel='Simulation time (s)',ylim=(0,30));ax[1].axhline(5,color='#555',ls='--',label='5 N');ax[1].set(xlim=(2.68,2.94),xlabel='Simulation time (s), impact detail',ylabel='Normal contact (N)');ax[1].legend()
for b in ax:b.grid(alpha=.2)
fig.savefig(o/'force.png',dpi=160)
msg=f"""## 05 天然物敲击：固定初态通过
hand_111 左手，Newton SolverMuJoCo CPU，960 Hz。自由复合刚体锤石，固定硬壳果目标置于砧座，测碰撞不模拟破壳。初态手指位于握持位；只控制手关节，锤石不绑定。每步至少两根不同手指对石各有 >0.01 N 法向载荷且石不靠外物，连续持握 {hold:.3f} s 后首次锤石–目标接触（{tf:.4f} s）；触前锤石鼻部竖直速度 {speed:.3f} m/s。峰值 {peak:.2f} N（{tp:.4f} s），为该步锤石对目标各接触点法向力之和，正例手碰目标峰值 {H.max():.2f} N。峰值时手对锤石有载部件：{', '.join(links)}。空手对照实际命中同一目标，手–目标峰值 {bare:.2f} N，锤石–目标始终 0 N，按同一判据不通过。仿真接触峰值依赖步长和接触参数，非实物标定。

证据：outputs/nov20/strike05/ 下 hit.mp4、bare.mp4（原速后附慢放）、force.png、hit_contacts.csv、bare_contacts.csv、hit_trace.csv、bare_trace.csv、XML/NPZ。trace 包含各指及末节载荷；contacts 含指节 ID、接触点、碰撞体和法向力。scripts/nov20/strike05.py 分别运行 bare、hit 后运行 report05.py。
"""
(o/'RESULTS.md').write_text(msg);print(msg)
