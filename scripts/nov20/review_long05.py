
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,subprocess,shutil
o=OUT/'strike05_long';tag=sys.argv[1];z=np.load(o/(tag+'.npz'));q=z['q'];a=z['a'];u=z['audit'];t=a[:,0];j=np.where(a[:,1]>.001)[0][0];k=a[:,1].argmax();b=np.load(o/'bare.npz')['a'];old=np.load(o/'old.npz')['audit'];hold=((a[:,5:10]>.01).sum(1)>=2)&(a[:,3]<.001);i=j-1
while i>=0 and hold[i]:i-=1
sec=(j-i-1)/960;assert sec>=2 and hold[k] and a[k,1]>=5 and a[:,2:4].max()==0 and u[:,5].max()==0 and u[:,1:3].min()>20 and b[:,2].max()>5 and b[:,1].max()==0
np.savetxt(o/'clearance.csv',u,delimiter=',',header='time_s,hand_target_mm,hand_anvil_mm,stone_target_overlap_mm,hand_stone_overlap_mm,non_nose_impact_N,hand_nose_mm',comments='',fmt='%.6f')
fig,ax=plt.subplots(3,1,figsize=(8,6),layout='constrained');ax[0].plot(t,u[:,1],label='hand / target');ax[0].plot(t,u[:,2],label='hand / anvil');ax[0].set_ylabel('Clearance (mm)');ax[0].legend()
for h,y,l in [(1,a[:,1],'Nose on target (N)'),(2,u[:,3],'Overlap (mm)')]:
 ax[h].plot(t,y);ax[h].set_xlim(t[j]-.025,t[j]+.08);ax[h].set_ylabel(l)
ax[2].set_xlabel('Simulation time (s)');fig.savefig(o/'audit.png',dpi=150)
m=mj.MjModel.from_xml_path(str(o/(tag+'.xml')));d=mj.MjData(m);m.vis.global_.offwidth=700;m.vis.global_.offheight=460;ren=mj.Renderer(m,440,640);font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),20)
def draw(h):
 # Solver force and geom transforms refer to the pre-integration collision query.
 v=q[h-1] if h else np.load(o/'p-.095.npz')['q'][960];d.qpos[:]=np.r_[v[:30],v[[33,30,31,32]]];mj.mj_forward(m,d)
 err=np.linalg.norm(d.geom_xpos[m.geom('nose').id]-a[h,-3:])*1000
 im=Image.new('RGB',(1280,524),'#222a30')
 for jj,(az,el,dist,ctr) in enumerate([(-115,-6,.25,[-.045,-.002,.068]),(-67,-3,.146,[-.068,.009,.054])]):
  cam=mj.MjvCamera();cam.lookat[:]=ctr;cam.azimuth=az;cam.elevation=el;cam.distance=dist;ren.update_scene(d,camera=cam);im.paste(Image.fromarray(ren.render()),(640*jj,40))
 dr=ImageDraw.Draw(im);dr.text((12,7),'05 | Grip and striking end',font=font);dr.text((652,7),'Contact close-up  |  brown: target',font=font);dr.text((14,489),f't = {t[h]:.4f} s     Nose-target: {a[h,1]:.2f} N     Hand-target gap: {u[h,1]:.1f} mm     Solver overlap: {u[h,3]:.3f} mm',font=font);return im,err
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x524','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/'operation.mp4')],stdin=subprocess.PIPE);errors=[]
for h in list(range(0,len(t),32))+list(range(j-35,j+100)):
 im,err=draw(h);errors.append(err);proc.stdin.write(im.tobytes())
 if h==k:im.save(o/'impact.jpg')
proc.stdin.close();assert proc.wait()==0;ren.close();assert max(errors)<.15,max(errors)
msg=f"""## 05 长形锤石：穿模与击手间距复核（同编号替换预览）
旧版重放：手–目标全程最小 {old[:,1].min():.2f} mm，石–目标最大求解器重叠 {old[:,3].max():.3f} mm，初始化手–石最大重叠 {old[:,4].max():.2f} mm。旧目标和静态砧座沿竖直方向也有 4 mm 几何交叠。

保留 hand_111 和后部握持区；锤头中心沿石体长向外移 32 mm，以圆滑过渡体连接；目标底部与砧面相切。从同模型自然稳定后的持握姿态开始，在 960 Hz Newton 中仅写手关节。实际连续握持 {sec:.3f} s 后命中，仅前端对目标承载，峰值 {a[k,1]:.2f} N。全过程手–目标、石–砧座/地面均无有载接触。手–目标最小净距 {u[:,1].min():.2f} mm，手–砧座 {u[:,2].min():.2f} mm；石–目标最大重叠 {u[:,3].max():.3f} mm、手–石最大重叠 {u[:,4].max():.3f} mm。接触容许微小压入，并非严格零穿透。空手对照手–目标 {b[:,2].max():.2f} N，石–目标 0 N，判失败。视频按碰撞检测时刻对齐姿态，抽取视频帧的石端中心与求解器坐标最大差 {max(errors):.4f} mm。
证据 outputs/nov20/strike05_long/operation.mp4、impact.jpg、audit.png、clearance.csv、{tag}_contacts.csv、{tag}_trace.csv。固定初态仿真，非真人安全评估。本月仍为 20 个独立场景。
"""
(o/'RESULTS.md').write_text(msg);print(msg)
