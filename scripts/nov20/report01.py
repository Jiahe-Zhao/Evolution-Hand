
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import csv,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
o=OUT/'branch01';sa={};fig,ax=plt.subplots(3,1,figsize=(8,6),sharex=True,layout='constrained');font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),19)
for tag in ['touch','grip']:
 z=np.load(o/(tag+'.npz'));a=z['a'];q=z['q'];t=a[:,0];s=t>=1.35;take=t>=.70;ax[0].plot(t,a[:,4],label=tag+' branch reaction');ax[1].plot(t[take],a[take,3],label=tag);sa[tag]=a
 if tag=='grip':
  assert np.ptp(t[s])>=2 and a[take,3].max()<10 and np.allclose(a[s,1],2) and np.all(a[s,4]>2) and (a[s,7:11]>.02).sum(1).min()>=2
  ax[0].plot(t,a[:,1],'--',label='external load');ax[1].axhline(10,color='k',ls=':');ax[2].plot(t,a[:,11],label='thumb tip');ax[2].plot(t,a[:,9],label='ring, all segments');ax[2].plot(t,a[:,10],label='little, all segments')
 else:assert a[:,5:11].max()>.01 and a[-1,3]>10 and abs(a[-1,4])<.01
 # Verify no wrist joint reaches its mechanical limit in the accepted segment.
 m=mj.MjModel.from_xml_path(str(o/(tag+'.xml')));d=mj.MjData(m)
 if tag=='grip':assert ((q[take,:6]>m.jnt_range[:6,0]+.005)&(q[take,:6]<m.jnt_range[:6,1]-.005)).all()
 m.vis.global_.offwidth=640;m.vis.global_.offheight=400;ren=mj.Renderer(m,400,640);cams=[]
 for az,el in [(130,-12),(85,12)]:
  cam=mj.MjvCamera();cam.lookat[:]=[-.051,.013,.126];cam.distance=.25;cam.azimuth=az;cam.elevation=el;cams.append(cam)
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/(tag+'.mp4'))],stdin=subprocess.PIPE)
 for k in range(0,len(t),8):
  d.qpos[:]=q[k];mj.mj_forward(m,d);im=Image.new('RGB',(1280,480),'#222a30')
  for j,cam in enumerate(cams):
   ren.update_scene(d,camera=cam);im.paste(Image.fromarray(ren.render()),(j*640,40))
  dr=ImageDraw.Draw(im);dr.text((12,8),f'01   {tag}   |   oblique',font=font);dr.text((653,8),'View along the branch',font=font);dr.text((12,450),f'{t[k]:.2f} s    |   added load {a[k,1]:.2f} N     branch reaction {a[k,4]:.2f} N     |     '+(f'palm displacement {a[k,3]:.1f} mm' if t[k]>=.7 else 'before load reference'),font=font)
  if tag=='grip' and k==960:im.resize((768,288)).save(o/'preview.jpg')
  proc.stdin.write(im.tobytes())
 proc.stdin.close();assert proc.wait()==0;ren.close()
for i,b in enumerate(ax):b.grid(alpha=.2);b.legend(fontsize=8);b.axvspan(1.35,sa['grip'][-1,0],alpha=.12,color='green');b.set_ylabel(['Force (N)','Palm slip (mm)','Normal load (N)'][i])
ax[2].set_xlabel('Simulation time (s)');fig.savefig(o/'load_slip.png',dpi=155)
a=sa['grip'];u=sa['touch'];s=a[:,0]>=1.35;ids=np.where(u[:,3]>10)[0];pos=np.array([-.053,0,.151]);ct=list(csv.DictReader(open(o/'grip_contacts.csv')));parts=sorted(set(x['body'] for x in ct if float(x['time_s'])>=1.35 and float(x['normal_N'])>.01))
msg=f"""## 01 抓枝承重：固定初态通过
hand_111 左手，Newton SolverMuJoCo CPU，480 Hz。固定水平圆枝直径 18 mm、长 180 mm，手枝滑动摩擦系数 0.8。两试次枝条、起始手位、重力和载荷程序相同。先定位，0.70 s 开始撤去腕部六轴驱动，0.85 s 起六轴均无主动驱动；绕枝调整了起始握姿，使腕部接近枝条下方。手指关节主动抓持。场内无地面、无手枝绑定；满载期间腕部未触关节限位。掌部施加竖直向下外力，0.85–1.35 s 从 0 增至 2.00 N，随后恒定；另有自重。

仅触枝、不闭合包握：实测手指法向载荷（单指合计）峰值 {u[:,6:11].max():.2f} N；{u[ids[0],0]:.3f} s 时掌相对枝位移首次超过 10 mm，末帧失去枝条接触，提前终止，未到满载。
包握后：满载持续 {np.ptp(a[s,0]):.3f} s；以撤托开始时的掌根点为基准，至结束三维位移最大 {a[a[:,0]>=.70,3].max():.2f} mm（不是只算沿枝轴向）。满载枝条向上反力均值 {a[s,4].mean():.2f} N；含重力和外加载荷。满载实际有载部件：{', '.join(parts)}。拇指末节平均法向力 {a[s,11].mean():.2f} N；无名指、小指各节合计均值 {a[s,9].mean():.2f} / {a[s,10].mean():.2f} N。逐接触表含所有手指指节、接触点、法向力及枝对手世界坐标力，不把中近节误记为指尖。此项验证固定初态的 2 N 附加载荷，不是最大承重或随机成功率。

证据：outputs/nov20/branch01/ 下 grip.mp4、touch.mp4（半速）、load_slip.png、grip_contacts.csv、touch_contacts.csv、grip_trace.csv、touch_trace.csv 及 XML/NPZ；scripts/nov20/branch01.py grip / touch 后运行 report01.py。
"""
(o/'RESULTS.md').write_text(msg);print(msg)
