
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
import csv,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=OUT/'clip22';Z={k:np.load(o/(k+'.npz')) for k in ('hand','none','left','right','both')};h=Z['hand']['a'];t=h[:,0];q=Z['hand']['q'];O=q[0,27:30]-h[0,1:4]/1000
C=list(csv.DictReader(open(o/'hand_contacts.csv')));last=max(float(z['time_s']) for z in C);free=t>last
assert free.sum()>=240 and abs(h[-192:,4:6]).max()<.2
good=(h[:,4:6].min(1)>2.05)&(h[:,6]>.01)&(h[:,10]>.01)&(h[:,[7,8,9,11]].max(1)==0)
start=np.flatnonzero(t>=.62)[0];i15=np.flatnonzero((t>=.62)&(h[:,3]-h[start,3]>=15))[0];end=np.flatnonzero(t<=1.32)[-1];assert good[start:end+1].all() and i15<end
# Actual bottom corners of the rotated guide shank, relative to socket mouth.
corners=np.array([[x,y,-.045] for x in [-.0068,.0068] for y in [-.0081,.0081]])
clear=np.array([1000*((R.from_quat(x[30:34]).apply(corners)+x[27:30])[:,2].min()-O[2]+.0155) for x in q]);assert clear[end]>0
ret=np.flatnonzero(free & (abs(h[:,4:6]).max(1)<=.2));ret=next(j for j in ret if (abs(h[j:,4:6]).max()<=.2))
modes=['none','left','right'];title=['No squeeze','Left only','Right only'];peaks=[]
for i,k in enumerate(modes):
 a=Z[k]['a'];assert a[:,3].max()<2 and a[-100:,4:6].max()<.2
 idx=(a[:,0]>=.6)&(a[:,0]<=1.2)
 if i==0:assert (a[idx,12:14].min(0)>.01).all()
 else:assert a[idx,4+i-1].min()>2.05 and a[idx,4+(2-i)].max()<2.05 and a[idx,12+(2-i)].min()>.01
 peaks.append(a[:,3].max())
assert Z['both']['a'][-1,3]>=20
fig,ax=plt.subplots(3,1,figsize=(8.6,7.1),sharex=True)
ax[0].plot(t,h[:,4],label='Left ear / thumb',c='#c16c15');ax[0].plot(t,h[:,5],label='Right ear / index',c='#197d60');ax[0].axhline(2.05,ls=':',c='.45');ax[0].set_ylabel('Ear travel (mm)')
ax[1].plot(t,h[:,3],label='Hand-operated plug',c='#205f83');ax[1].axhline(15,ls='--',c='.55');ax[1].set_ylabel('Axial position (mm)')
for k,label in zip(modes,title):a=Z[k]['a'];ax[2].plot(a[:,0],a[:,3],label=label)
ax[2].axvspan(.58,1.28,color='.9',zorder=-2);ax[2].set(ylabel='Control pull (mm)',xlabel='Time (s)')
for a in ax:a.legend(loc='upper left',fontsize=8,ncol=3);a.grid(alpha=.18)
for a in ax[:2]:a.axvspan(t[start],t[end],color='#c6e5d0',alpha=.35,zorder=-3)
fig.tight_layout();fig.savefig(o/'curves.png',dpi=165)
font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),16)
def render(mode):
 neg=mode=='controls';data=Z['hand'];vm=mj.MjModel.from_xml_path(str(o/'hand.xml'));vm.vis.global_.offwidth=800;vm.vis.global_.offheight=480;vm.vis.headlight.ambient[:]=.42
 for i in range(vm.nlight):vm.light_castshadow[i]=0
 vd=mj.MjData(vm);W,H=(420,350) if neg else (640,440);ren=mj.Renderer(vm,H,W);chan=3 if neg else 2;imgsize=(W*chan,H+76)
 for i in range(vm.ngeom):
  if vm.geom(i).name.startswith('handgeom'):vm.geom_group[i]=5
 vis=mj.MjvOption();vis.geomgroup[5]=1;cut=mj.MjvOption();cut.geomgroup[5]=0;hide=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith(('guideX','guideY','wall','rail'))];rgba=vm.geom_rgba.copy()
 p=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{imgsize[0]}x{imgsize[1]}','-r','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(o/(mode+'.mp4'))],stdin=subprocess.PIPE)
 for j in range(0,len(Z['none' if neg else 'hand']['q']),16):
  im=Image.new('RGB',imgsize,'#172b32');dr=ImageDraw.Draw(im)
  for ii,k in enumerate(modes if neg else ['hand','hand']):
   x=Z[k]['q'][j];a=Z[k]['a'][j];vd.qpos[:27]=x[:27];vd.qpos[27:34]=x[[27,28,29,33,30,31,32]];vd.qpos[34:]=x[34:];mj.mj_forward(vm,vd);sub=neg or ii==1;cam=mj.MjvCamera();cam.lookat[:]=.65*x[27:30]+.35*O+[0,0,-.016];cam.distance=.151 if sub else .192;cam.azimuth=4 if sub else 42;cam.elevation=-9 if sub else -26
   vm.geom_rgba[:]=rgba
   if sub:vm.geom_rgba[hide,3]=.12
   ren.update_scene(vd,cam,scene_option=cut if sub else vis);im.paste(Image.fromarray(ren.render()),(ii*W,36))
   dr.text((ii*W+8,7),title[ii]+'  |  same pull' if neg else ('hand_111  |  thumb + index' if not sub else 'Mechanism view  |  hand hidden, socket walls transparent'),font=font,fill='white');dr.text((ii*W+8,H+44),f'{a[0]:.2f}s    L {a[4]:.1f}   R {a[5]:.1f}    Pull {a[3]:.1f} mm',font=font,fill='white')
  if not neg and j==960:im.save(o/'preview.jpg')
  p.stdin.write(im.tobytes())
 p.stdin.close();assert p.wait()==0;ren.close()
render('operation');render('controls')
doc=f"""## 22 捏合双耳卡扣并拔出：固定初态通过（2026-11-17）

独立新增场景，03 和 18 继续暂缓。插座固定，插头六自由度；两侧卡舌各有被动直线滑块、90 N/m 回弹弹簧、0.65 N·s/m 阻尼、3.8 mm 名义行程与端部弹性限位。该模型等效表示侧耳的横向弹性，不模拟悬臂梁的局部应变。每侧倒扣与插座肩台横向重叠 2.05 mm；只有实体碰撞阻挡拔出，未按任务阶段开关约束或碰撞。

同一插座、相同初态和轴向拉力历程：拉力从 0 升至 0.50 N，并保持 0.70 s 后卸载。独立夹具对指定耳施加 0.30 N 横向力；手移开。未捏/只捏左/只捏右的最大轴向位移分别 **{peaks[0]:.2f} / {peaks[1]:.2f} / {peaks[2]:.2f} mm**。只捏一侧时被压侧从肩台退开，另一侧仍有持续有载倒扣接触；解除横向力后两耳均回到零位附近。双侧夹具同拉力正对照通过 20 mm 即停测。

现有 hand_111 左手，Newton SolverMuJoCo CPU，960 Hz。正例只指令手关节；弹簧是被动力，插头无附加拔力。拇指末节压左耳，食指末节压右耳。自 {t[start]:.3f} 至 {t[end]:.3f} s，逐步验证双耳均内移 >2.05 mm、两根指定指分别承载 >0.01 N，无错指/其他手部对耳垫、凸缘或臂的有载接触。该段轴向净拔出 **{h[end,3]-h[start,3]:.2f} mm**，{t[i15]:.3f} s 跨越 15 mm；段末插头导向柄底部最低角比插座口高 {clear[end]:.2f} mm。该段双耳最小行程 {h[start:end+1,4].min():.2f} / {h[start:end+1,5].min():.2f} mm，拇指/食指最小载荷 {h[start:end+1,6].min():.3f} / {h[start:end+1,10].min():.3f} N。全程两侧峰值行程 {h[:,4].max():.2f} / {h[:,5].max():.2f} mm。

插头拔离后移至接料台并松手；最后有载手–插头接触 {last:.3f} s。自 {t[ret]:.3f} s 至末帧，左右行程均在零位 ±0.2 mm 内（持续 {t[-1]-t[ret]:.3f} s）。从插头已扣入、手在捏持位附近开始，不代表随机初态成功率。

证据：outputs/nov20/clip22/ 下 operation.mp4、controls.mp4（均半速、画面时间为物理时间）、curves.png、hand_contacts.csv、hand_trace.csv、none/left/right/both 的 trace.csv、各次 XML/NPZ。运行 scripts/nov20/clip22.py none / left / right / both / hand，再运行 report22.py。
"""
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';old=f.read_text()
if doc.splitlines()[0] not in old:f.write_text(old.rstrip()+'\n\n'+doc)
print(doc)
