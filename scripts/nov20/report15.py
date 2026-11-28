
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from nut15 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import csv,subprocess
f=np.load(out/'hand.npz');x=f['data'];t=x[:,0];a=x[:,1];feed=x[:,2]*1000;pred=a*pitch*1000/(2*np.pi);wr=np.rad2deg(R.from_euler('XYZ',x[:,4:7]).magnitude())
push=np.load(out/'push.npz')['data'];tor=np.load(out/'torque.npz')['data'];low=np.load(out/'subtorque.npz')['data'];turns=a[-1]/(2*np.pi);err=np.max(np.abs(feed-pred));fac=f['faces']
# Require an exclusive, forward, force-producing episode, rather than just a touch.
F=x[:,7:12];tau=x[:,12:17]*1000;rotating=np.gradient(a)>0;ep=[]
for finger in (1,2):
 mask=(F[:,finger]>.02)&(F[:,3-finger]<.01)&(tau[:,finger]>.3)&rotating
 for i,j in np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2):
  da=np.rad2deg(a[j-1]-a[i])
  if j-i>=10 and da>=5:ep.append([t[i],t[j-1],finger+1,da,float(np.mean(F[i:j,finger])),float(np.mean(tau[i:j,finger])),hex(int(np.bitwise_or.reduce(fac[i:j,finger])))])
ep.sort();group=[]
for row in ep:
 if not group or group[-1][2]!=row[2]:group.append(row)
 elif row[3]>group[-1][3]:group[-1]=row
assert turns>=2 and feed[-1]>=3.5 and err<.02 and wr.max()<5 and len(group)>=3 and push[:,2].max()*1000<.02 and np.max(F[-120:])<.01
with (out/'handoffs.csv').open('w') as fp:
 w=csv.writer(fp);w.writerow(['start_s','end_s','digit_2index_3middle','net_deg','mean_normal_N','mean_clockwise_moment_mNm','sector_mask_hex']);w.writerows(ep)
np.savetxt(out/'trace.csv',np.c_[t,np.rad2deg(a),feed,pred,wr,F,tau,fac],delimiter=',',header='time_s,net_deg,feed_mm,lead_prediction_mm,wrist_deg,'+','.join(f'{k}_{s}' for k in ['load_N','moment_mNm','sector_mask'] for s in ['thumb','index','middle','ring','little']),comments='',fmt='%.8g')
fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True);axs[0].plot(t,np.rad2deg(a),color='#355a77');axs[0].axhline(720,color='#888',ls='--');axs[0].set_ylabel('Nut angle (deg)');axs[1].plot(t,feed,color='#da744a',lw=2,label='Measured body position');axs[1].plot(t,pred,'k--',lw=.8,label='Pitch x net turns');axs[1].axhline(3.5,color='#888',ls=':');axs[1].set_ylabel('Axial feed (mm)');axs[1].legend();axs[2].plot(t,F[:,1],label='Index',color='#307a70');axs[2].plot(t,F[:,2],label='Middle',color='#9e5882');axs[2].set_ylabel('Finger load (N)');axs[2].set_xlabel('Time (s)');axs[2].legend()
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Threaded hex nut | 1.75 mm lead per turn | Newton');fig.tight_layout();fig.savefig(out/'angle_feed.png',dpi=155);plt.close(fig)
vm=mj.MjModel.from_xml_path(str(out/'nut15.xml'));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];opt=mj.MjvOption();opt.geomgroup[5]=0;cams=[]
for target,d,az,el in [([cx+.014,cy,z+.018],.225,108,-37),([cx,cy,z-.003],.095,130,-31)]:
 cam=mj.MjvCamera();cam.lookat[:]=target;cam.distance=d;cam.azimuth=az;cam.elevation=el;cams.append(cam)
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(out/'operation.mp4')],stdin=subprocess.PIPE)
names=f['names'];starts=f['starts'];qs=f['q']
for j in range(0,len(qs),8):
 for k,name in enumerate(names):
  i=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,str(name))
  if i>=0:vd.qpos[vm.jnt_qposadr[i]]=qs[j,starts[k]]
 mj.mj_forward(vm,vd);ren.update_scene(vd,cams[0]);left=ren.render().copy();vm.geom_group[hs]=5;ren.update_scene(vd,cams[1],scene_option=opt);right=ren.render().copy();vm.geom_group[hs]=0;i=2*j;im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,24],fill=(24,34,38));dr.text((6,5),f'{t[i]:.2f}s | index {F[i,1]:.2f} N | middle {F[i,2]:.2f} N',fill='white');dr.text((486,5),f'Nut only | {a[i]/(2*np.pi):.2f} turns | feed {feed[i]:.2f} mm',fill='white');proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close();assert proc.wait()==0;ren.close()
txt=f"""## 15 旋拧螺帽：已啮合起始、旋转阻力与换指

固定竖直螺柱、带贯通孔及倒角的六角螺帽（对边约 29.8 mm，厚 10 mm）；标记面用于读角度。初态螺柱顶端已进入螺帽 1.75 mm，即一扣。单头右旋螺旋副：顺时针 1 圈沿世界 −Z 进给 1.75 mm。转动、轴移均为被动关节，以求解器等式耦合；实测进给取螺帽刚体世界高度变化，未用转角代替位移。起始啮合不计入旋进量。关节库仑阻力矩 1.60 mN·m，黏性阻尼 0.14 mN·m·s/rad。阻力为设定量，非实物标定。未建真实牙面、起扣或错牙。

同模型 Newton SolverMuJoCo CPU，480 Hz，无手对照各 1.5 s，前 0.15 s 线性加载：
- 直接向下推 3 N、不施扭矩：最大轴移 {1000*max(abs(push[:,2])):.6f} mm；最大转角 {np.rad2deg(max(abs(push[:,1]))):.4f}°，未滑下一扣。限于本载荷，不是任意轴力下均自锁。
- 顺时针 0.50 mN·m：最大转角 {np.rad2deg(max(abs(low[:,1]))):.4f}°；3.00 mN·m：净转 {tor[-1,1]/(2*np.pi):.3f} 圈，进给 {tor[-1,2]*1000:.3f} mm。

hand_111 左手；仅写手关节目标，正例不给螺帽施加外力或位姿。食指、中指沿相反侧轮换拨转，回程指离开后重新着力。腕三转动关节各设 ±2.5° 软限位，目标均为中立；实测腕总偏转最大 {wr.max():.3f}°。截止回合末净转 **{turns:.4f} 圈（{np.rad2deg(a[-1]):.2f}°）**，刚体实测进给 **{feed[-1]:.5f} mm**；全程螺距关系最大残差 {err:.6f} mm。末 0.25 s 无指载，结果仍保留。**满足净转 ≥2 圈、进给 ≥3.50 mm、腕受限且至少一次换指。**

换指记录取实际接触载荷，不取动作指令：单指法向载荷 >0.02 N、另一指 <0.01 N，接触力对螺柱轴的顺时针力矩 >0.3 mN·m，连续至少 10 个物理步、同时正转 ≥5°。将连续同指区间合并用于计数，取得 {len(group)-1} 次独立的指间切换。下表为相邻不同指的代表段，并非全部转角分摊。
|时段 s|着力指|该段实际正转 °|平均顺时针力矩 mN·m|
|---|---|---:|---:|
"""
for t0,t1,k,deg,force,mom,mask in group:txt+=f"|{t0:.3f}–{t1:.3f}|{['食指','中指'][k-2]}|{deg:.2f}|{mom:.2f}|\n"
txt+="\n证据：outputs/nov20/nut15/operation.mp4、angle_feed.png、handoffs.csv、trace.csv、hand.npz（真实关节动作及接触面掩码）、push/torque/subtorque.npz、nut15.xml。scripts/nov20/nut15.py 分别运行 push / subtorque / torque / hand，再用 scripts/nov20/report15.py 复核。固定初态正例，不代表随机成功率。\n"
p=OUT.parent.parent/'docs/SCENES_20_20261116.md';s=p.read_text()
if txt.splitlines()[0] not in s:p.write_text(s.rstrip()+'\n\n'+txt)
(out/'RESULTS.md').write_text(txt);print(txt)
