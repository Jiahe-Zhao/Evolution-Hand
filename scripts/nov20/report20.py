
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
import csv,json,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
o=OUT/'scan20';data={k:np.load(o/(k+'.npz')) for k in ['idle','hand','blocked']};h=data['hand']['a'];i=data['idle']['a'];b=data['blocked']['a'];e,=np.flatnonzero(h[:,9]);row=h[e];t=row[0];valid=(i[:,1]>=80)&(i[:,1]<=200)&(i[:,2]<=10)&(i[:,3]>0);assert valid.sum()>480 and not i[:,8:].any() and b[:,8].sum()==1 and not b[:,9].any() and b[b[:,8]>0,3].max()==0
reads=json.loads((o/'hand_reads.json').read_text());assert reads==[[t,'73513537']] and h[h[:,0]>=.2,7].min()>=2 and row[5]>.01 and h[:,6].max()==0 and h[-1,4]<.6
contacts=list(csv.DictReader(open(o/'hand_contacts.csv')));assert not any(r['object']=='external_support' for r in contacts);at=[r for r in contacts if abs(float(r['time_s'])-t)<.0001];print(row,at)
trig=[r for r in contacts if r['object']=='paddle']
with open(o/'trigger_contacts.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=trig[0].keys());w.writeheader();w.writerows(trig)
parts='/'.join(sorted(set(r['hand_link'] for r in at if r['object']=='paddle')))
support=', '.join(sorted(set(r['hand_link'] for r in at if r['object'] in ('grip','heel','housing','bridge') and not r['hand_link'].startswith('link_2_'))))
back=np.flatnonzero((h[:,0]>t)&(h[:,4]<=.6))[0];fig,axs=plt.subplots(3,1,figsize=(8,6),sharex=True)
for ax,n,lab,band in zip(axs,[1,2,4],['Window to code (mm)','Optical axis error (deg)','Trigger travel (mm)'],[(80,200),(0,10),(3,6)]):
 ax.axhspan(*band,color='#daeade',alpha=.65);ax.plot(i[:,0],i[:,n],'--',label='Aim only',color='gray');ax.plot(h[:,0],h[:,n],label='Index pull',color='#155575');ax.axvline(t,color='#bd651b',ls=':');ax.scatter([t],[row[n]],color='#bd651b',s=17,zorder=4);ax.set_ylabel(lab);ax.grid(alpha=.17)
axs[0].legend(ncol=2);axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(o/'curves.png',dpi=150)
vm=mj.MjModel.from_xml_path(str(o/'hand.xml'));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,504);B=np.loadtxt(o/'board.txt')[0];poses=data['hand']['poses'];qs=data['hand']['q'];frames=[]
for j in range(0,len(qs),16):
 q=qs[j];vd.qpos[:27]=q[:27];vd.qpos[27:34]=q[[27,28,29,33,30,31,32]];vd.qpos[34]=q[34];mj.mj_forward(vm,vd);cam=mj.MjvCamera();cam.lookat[:]=[-.094,-.005,.068];cam.distance=.32;cam.azimuth=128;cam.elevation=-30;ren.update_scene(vd,cam)
 if e<=j<e+80:
  gg=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(gg,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.18,.95,.30,1],np.float32));mj.mjv_connector(gg,mj.mjtGeom.mjGEOM_CAPSULE,.0005,poses[j,:3],B);ren.scene.ngeom+=1
 left=ren.render().copy();cam.lookat[:]=poses[j,6:9];cam.distance=.145;cam.azimuth=95;cam.elevation=-13;ren.update_scene(vd,cam);im=Image.fromarray(np.hstack([left,ren.render()]));dr=ImageDraw.Draw(im);dr.rectangle((0,0,1008,26),fill='#172f39');dr.text((8,7),f'hand_111 | {h[j,1]:.1f} mm   {h[j,2]:.1f} deg | '+('EAN-8: 73513537' if j>=e else 'waiting for trigger'),fill='white');dr.text((511,7),f'Index + trigger | {h[j,4]:.2f} mm   {h[j,5]:.2f} N   t={h[j,0]:.2f}s',fill='white');frames.append(np.asarray(im))
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1008x360','-r','30','-i','-','-pix_fmt','yuv420p',str(o/'operation.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()
doc=f'''## 20 持扫描器读码：固定初态通过（2026-11-17）

hand_111 左手，Newton SolverMuJoCo CPU，480 Hz。从已握持初态抬腕后扣动；扫描器自由，不固定到手部。约 85 g，握柄为直径 34 mm 的倾斜胶囊体，带端部和握柄到机头的连接件。食指扳机被动直线导向，名义行程 6 mm，回弹刚度 130 N/m，阻尼 0.8 N·s/m。3 mm 上升沿触发，释放到 0.6 mm 以下复位；只有上升沿尝试读码。

固定 EAN-8 码面（单模块 0.62 mm）。读区按窗口到码心 80–200 mm、光轴与中心连线 ≤10° 定义。测试中心射线遮挡，再从窗口对已知码区的 67 个模块和两侧静区分别投射射线，要求各采样在光轴 15° 半角内，按第一个命中物判定黑白条，校验护栏、数字和校验位。这是理想几何读头，不是摄像头图像识别。

|试次|实测|
|---|---|
|仅对准、不扣动|满足距离、角度且无遮挡 {valid.sum()/480:.3f} s；无扳机触发，读码 0 次。|
|食指扣动|{t:.4f} s 读出 **{reads[0][1]}**，只读 1 次。窗口到码心 **{row[1]:.2f} mm**，光轴误差 **{row[2]:.2f}°**，视线无遮挡；行程 **{row[4]:.2f} mm**，食指法向力 **{row[5]:.3f} N**，接触部件 {parts}。触发时其他手指/手掌扳机载荷为零。|
|遮住码面再扣动|相同扣动命令、仍跨越触发行程，中心射线先命中遮挡片，读码 0 次。|

正例触发时支持握持的手部件：{support}。未记录机身与地面/码架的承重接触。扳机全程最大实测行程 {h[:,4].max():.2f} mm，松指后在 {h[back,0]:.3f} s 首次回到释放阈值，终值 {h[-1,4]:.2f} mm（零点附近的有符号值）。视频绿线是读码事件标注；未验证随机初态或桌面拾取。

证据：outputs/nov20/scan20/ 下 operation.mp4、curves.png、trigger_contacts.csv、hand_trace.csv、hand_contacts.csv、三次试次的 XML/NPZ/reads.json 和固定 board.txt。复现：在 scripts/nov20 用 IsaacLab3 Python 依次运行 scan20.py idle / hand / blocked，report20.py 复核。
'''
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';old=f.read_text();title=doc.splitlines()[0];f.write_text(old.rstrip()+'\n\n'+doc) if title not in old else None;print(doc)
