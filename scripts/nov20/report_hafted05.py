
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
import matplotlib,csv,subprocess,json,hashlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=OUT/'hafted05_verify';tag='h.010';z=np.load(o/(tag+'.npz'));a=z['a'];q=z['q'];t=a[:,0];bare=np.load(o/'bare.npz');ba=bare['a'];good=((a[:,9:14]>.01).sum(1)>=2)&(a[:,3]<.001)&(a[:,4]<.001)
j=np.flatnonzero(a[:,1]>.001)[0];i=j-1
while i>=0 and good[i] and a[i,1]<.001:i-=1
duration=(j-i-1)/960;peak=a[:,1].argmax();assert duration>=2 and good[peak] and a[peak,1]>=5 and a[:,2:5].max()==0 and ba[:,1].max()==0 and ba[:,2].max()>=5
names=[x.split('/')[-1] for x in z['names']];hand=names.index('link_0_0');hp=z['pose'][:,hand];op=z['pose'][:,-1];rel=R.from_quat(hp[:,3:]).inv().apply(op[:,:3]-hp[:,:3]);stable=(t>=1.2)&(t<=3.3);drift=np.linalg.norm(rel[stable]-rel[stable][0],axis=1).max()*1000;assert good[stable].all()
fig,ax=plt.subplots(3,1,figsize=(9,7),layout='constrained')
ax[0].plot(t,a[:,1],label='stone - target');ax[0].plot(ba[:,0],ba[:,2],label='bare hand - target');ax[0].axhline(5,color='gray',ls='--');ax[0].set(xlim=(3.27,3.65),ylabel='Impact normal load (N)');ax[0].legend()
for k,n in enumerate(['thumb','index','middle','ring','little']):ax[1].plot(t,a[:,9+k],label=n)
ax[1].axvspan(1.2,3.3,alpha=.15,color='green');ax[1].set_ylabel('Hand on handle (N)');ax[1].legend(ncol=5,fontsize=8)
ax[2].plot(t,a[:,7],label='hand-target gap (mm)');ax[2].set(ylabel='Clearance (mm)',xlabel='Simulation time (s)');ax[2].legend()
for h in ax:h.grid(alpha=.2)
fig.savefig(o/'force_clearance.png',dpi=150)
font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),20);sync=[]
for name in [tag,'bare']:
 zz=np.load(o/(name+'.npz'));aa=zz['a'];qq=zz['q'];m=mj.MjModel.from_xml_path(str(o/(name+'.xml')));d=mj.MjData(m);m.vis.global_.offwidth=700;m.vis.global_.offheight=500;ren=mj.Renderer(m,440,640)
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x524','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(o/('operation.mp4' if name==tag else 'control.mp4'))],stdin=subprocess.PIPE)
 full=list(range(0,len(aa),32));slow=list(range(int(3.28*960),int(3.62*960),2));seq=full+slow
 for frame,k in enumerate(seq):
  v=qq[k-1] if k else np.load(o/'g-.008_.080.npz')['q'][1920];d.qpos[:27]=v[:27]
  if name==tag:d.qpos[27:]=np.r_[v[27:30],v[[33,30,31,32]]]
  mj.mj_forward(m,d);im=Image.new('RGB',(1280,524),'#222a30')
  for jj,(az,el,dist,ctr) in enumerate([(-45,-17,.34,[-.006,-.067,.122]),(-110,-9,.235,[.003,-.10,.10])]):
   cam=mj.MjvCamera();cam.lookat[:]=ctr;cam.azimuth=az;cam.elevation=el;cam.distance=dist;ren.update_scene(d,camera=cam);im.paste(Image.fromarray(ren.render()),(640*jj,40))
  dr=ImageDraw.Draw(im);dr.text((12,8),('05 | Rough stone + short wood handle' if name==tag else '05 | Bare-hand negative control'),font=font);dr.text((740,8),'1/16x impact replay' if frame>=len(full) else '1x simulation playback',font=font);dr.text((12,489),f'{aa[k,0]:.4f} s    Stone-target {aa[k,1]:.2f} N    Hand-target {aa[k,2]:.2f} N    Clearance {aa[k,7]:.1f} mm',font=font);proc.stdin.write(im.tobytes())
  if name==tag and abs(k-peak)<2:im.save(o/'impact.jpg')
  if name==tag and k==1920:im.save(o/'grip.jpg')
 proc.stdin.close();assert proc.wait()==0;ren.close()
rows=list(csv.DictReader(open(o/(tag+'_contacts.csv'))));parts=sorted({x['hand_link']+' / '+x['object_part'] for x in rows if abs(float(x['time_s'])-t[peak])<.0002})
metrics={'hold_before_first_impact_s':duration,'first_impact_s':float(t[j]),'peak_impact_s':float(t[peak]),'peak_impact_N':float(a[peak,1]),'hand_target_peak_N':float(a[:,2].max()),'other_support_peak_N':float(a[:,3].max()),'wood_target_peak_N':float(a[:,4].max()),'stone_target_overlap_mm':float(a[:,5].max()),'hand_hammer_overlap_mm':float(a[:,6].max()),'min_hand_target_gap_mm':float(a[:,7].min()),'stable_hold_drift_mm':float(drift),'negative_hand_target_peak_N':float(ba[:,2].max()),'asset_sha256':hashlib.sha256((OUT/'hafted05/asset.xml').read_bytes()).hexdigest()}
(o/'metrics.json').write_text(json.dumps(metrics,indent=2))
msg=f"""## 05 当前粗糙石锤＋短木柄：握持与敲击复验通过
当前已确认外形原样载入，不修改石头、木柄网格、尺寸和质量；无绑缚，头柄仍是同一自由组合刚体。hand_111 左手、Newton SolverMuJoCo CPU，960 Hz，只驱动手关节。先用自然接触稳定的握持状态作为正式回合初态，再抬手、移至目标上方、保持、敲击；不包含从台面拾取。无吸附、锤子额外驱动力或运动绑定。预备稳定过程不是验收计时。
正式回合首次石–目标接触前，至少两根不同手指各承担 >0.01 N 且无外物支撑，连续 {duration:.3f} s；其中 1.20–3.30 s 为定位后的持握段，锤头中心相对掌最大漂移 {drift:.3f} mm。首次命中 {t[j]:.4f} s；石头–目标峰值 {a[peak,1]:.2f} N（{t[peak]:.4f} s）。全程手–目标、木柄–目标、锤子–砧座/地面载荷均为0。手–目标最小净距 {a[:,7].min():.2f} mm。石–目标最大求解器重叠 {a[:,5].max():.3f} mm，手–锤最大 {a[:,6].max():.3f} mm，并非绝对零压入。峰值时持握接触：{'; '.join(parts)}。实际主要承载的是长指近中节，不把它们称作指尖；逐步表保留五指末节载荷。
同一模型目标的空手对照：手–目标峰值 {ba[:,2].max():.2f} N，石–目标为0，失败。对照移动手以实际命中目标，并非与正例完全相同腕轨迹。目标固定，只验证敲击接触，不模拟破碎。为固定初态物理正反例，不是跨初态成功率。
文件 outputs/nov20/hafted05_verify/operation.mp4、control.mp4（原速+1/16慢放）、force_clearance.png、metrics.json、h.010_contacts.csv、bare_contacts.csv、h.010_trace.csv、h.010.xml/npz。当前05复验通过，本月独立编号仍为20/20，本修订不重复计数。
复现 scripts/nov20/verify_hafted05.py：预备 ROT=1.57 GRIP=.055 CX=-.008 CZ=.080 BEND=1.35 LIFT=0 SHIFT=0 TX=-.25 DOWN=0 STEPS=2400，tag g-.008_.080；正式 WARM=g-.008_.080 WI=1920 BEND=1.35 TX=.010 TY=-.125 TZ=.045 DOWN=.050，tag h.010；空手增加 SHIFT=.092 DOWN=.125，tag bare。报告 report_hafted05.py。
"""
(o/'RESULTS.md').write_text(msg);print(msg)
