
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from audit21 import *
from PIL import Image,ImageDraw,ImageFont
import csv,sys,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
tag=sys.argv[1];globals().update(evaluate(tag));assert ok and not ex[0].any();time=a[:,0];tt=time[up];rims=rim.any(0);rem=np.flatnonzero(inside)
assert (np.abs(loc[0,:,0])<.018).all() and (np.abs(loc[0,:,1])<.016).all() and ((loc[0,:,2]>0)&(loc[0,:,2]<.018)).all()
arr=np.zeros_like(ex)
for j in ids:arr[:,j]=np.maximum.accumulate(ex[:,j]&inb[:,j]&on[:,j])
assert np.array_equal(arr[-1],bowl);settlet=time[exitidx[ids].max()];firstreturn=a[mx,0];flow=int(ex[mx].sum());rimout=int(rims[ids].sum());rimsteps=int(rim.any(1).sum())
np.savetxt(o/'pour_trace.csv',np.c_[time,a[:,1],ex.sum(1),inb.sum(1),arr.sum(1),a[:,4:8]],delimiter=',',header='time_s,tilt_deg,exited,inside_bowl,ever_contacted_bowl_inside,hand_grain_contacts,gripping_hand_links,cup_bowl_floor_contacts,grains_on_rim',comments='',fmt='%.6f')
with open(o/'grains.csv','w') as f:
 w=csv.writer(f);w.writerow(['grain_id','first_loaded_rim_contact_s','first_exit_s','exit_height_in_cup_mm','first_bowl_contact_inside_s','final']);
 for j in range(30):
  first=lambda v:round(time[np.flatnonzero(v)[0]],5) if v.any() else ''
  w.writerow([j,first(rim[:,j]),first(ex[:,j]),round(1000*loc[exitidx[j],j,2],3) if end[j] else '',first(arr[:,j]),'bowl' if bowl[j] else 'cup' if inside[j] else 'outside'])
fig,ax=plt.subplots(2,1,figsize=(8.2,5.3),sharex=True)
ax[0].plot(time,a[:,1],color='#1b7075');ax[0].axhline(5,ls=':',color='.45');ax[0].set_ylabel('Actual cup tilt (deg)');ax[1].step(time,ex.sum(1),where='post',label='Crossed outer lip',color='#ba6021');ax[1].step(time,arr.sum(1),where='post',label='Landed in bowl',color='#244e85',ls='--');ax[1].axhspan(9,11,color='#baad6d',alpha=.18);ax[1].set(ylabel='Grains',xlabel='Time (s)');ax[1].legend()
for axis in ax:axis.axvspan(tt,time[-1],alpha=.13,color='#348764');axis.axvline(tt,color='#348764',ls=':');axis.grid(alpha=.17)
fig.tight_layout();fig.savefig(o/'curves.png',dpi=150);plt.close()
vm=mj.MjModel.from_xml_path(str(o/(tag+'.xml')));vd=mj.MjData(vm);vm.vis.global_.offwidth=800;vm.vis.global_.offheight=480;vm.vis.headlight.ambient[:]=.4
for i in range(vm.nlight):vm.light_castshadow[i]=0
ren=mj.Renderer(vm,448,640);vis=mj.MjvOption();vis.geomgroup[5]=1;bare=mj.MjvOption();bare.geomgroup[5]=0
for i in range(vm.ngeom):
 if vm.geom(i).name.startswith('handgeom'):vm.geom_group[i]=5
seeds=[vm.geom('seed'+str(j)).id for j in range(30)]
transparent=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith(('cupX','cupY','bowlwall'))];cam=mj.MjvCamera();qu=z['q'];frames=[];font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),17)
for k in range(0,len(a),12):
 Q=qu[k];vd.qpos[:27]=Q[:27];free=Q[27:].reshape(-1,7);vd.qpos[27:]=free[:,[0,1,2,6,3,4,5]].ravel();mj.mj_forward(vm,vd);cam.lookat[:]=[-.058,-.021,.088];cam.distance=.292;cam.azimuth=35;cam.elevation=-33;ren.update_scene(vd,cam,scene_option=vis);left=ren.render().copy()
 cam.lookat[:]=[-.066,-.018,.084];cam.distance=.237;cam.azimuth=20;cam.elevation=-24;vm.geom_rgba[transparent,3]=.22;ren.update_scene(vd,cam,scene_option=bare);right=ren.render().copy();vm.geom_rgba[transparent,3]=1
 im=Image.new('RGB',(1280,504),'#172c32');im.paste(Image.fromarray(left),(0,30));im.paste(Image.fromarray(right),(640,30));dr=ImageDraw.Draw(im);dr.text((10,4),'hand_111 | free cup + fixed bowl',font=font,fill='white');dr.text((650,4),'Flow detail | hand hidden, walls transparent',font=font,fill='white');dr.text((10,480),f'{time[k]:.2f} s   |   Actual tilt {a[k,1]:.1f} deg',font=font,fill='white');dr.text((650,480),f'Exited {ex[k].sum():02d}     Landed {arr[k].sum():02d}     Loaded initially 30',font=font,fill='white');frames.append(np.asarray(im))
 if k==int(exitidx[ids].min()/12)*12+12:im.save(o/'preview.jpg')
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x504','-r','40','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(o/'operation.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()
doc=f"""## 21 定量倾倒谷粒：固定初态通过（2026-11-17）

替补新增独立任务，不恢复 03 搬运或 18 手机点选。hand_111 左手、Newton SolverMuJoCo CPU，480 Hz；从已握杯初态抬腕，只控制手指与腕关节。杯及 30 粒各为独立自由刚体，碗固定。杯内口 36 × 32 mm，底至沿中心 18 mm，沿边圆角由半径 1 mm 胶囊碰撞体构成；每粒为 5.0 × 4.4 × 4.0 mm 椭球、50 mg，显式给定三轴惯量。颗粒之间、杯壁/杯口、碗、手部均保留接触。碗底为圆盘，四周有十六块倾斜壁板。未标定真实谷物摩擦。

验收：目标 10，容许落碗 9–11；杯中余粒不算撒漏，意外落到杯碗外 ≤1 粒；回正后至少 0.5 s 无新出料。为防止在空中误计，按粒编号登记杯固连坐标下首次越过杯口外缘，再登记位于碗内且有实际碗接触的首次时刻；回合末复核每粒所在位置与稳定性。回正定义为峰值后实际杯轴与竖直夹角回到 ≤5° 并持续至结束。

实测：初装 30；最终 **碗内 {bowl.sum()}、杯内 {inside.sum()}、撒漏 {lost.sum()}**。杯子实测峰值 {a[mx,1]:.2f}°，峰值时已出 {flow} 粒；之后到回正前又出 {end.sum()-flow} 粒。最后出料 {settlet:.3f} s，{tt:.3f} s 回正，随后 **{time[-1]-tt:.3f} s 新出料 0 粒**；末帧 {a[-1,1]:.2f}°。已出料颗粒通过外沿时，其中心在杯底以上 {height.min()*1000:.2f}–{height.max()*1000:.2f} mm；{rimout}/{end.sum()} 粒检测到有载杯沿接触（无接触记录者不据此声称撞沿）。碗内谷粒最后 0.5 s 相对末帧最大位移 {stable*1000:.3f} mm。全程手–谷粒有载接触 0，杯–碗/地面有载接触 0，持续由手接触持杯；接触载荷记录阈值 10 µN。单次固定初态结果，不是随机装料成功率。

证据：outputs/nov20/pour21/ 下 {tag}.xml、{tag}.npz、operation.mp4、curves.png、pour_trace.csv、grains.csv。视频右侧隐藏手并将壁板透明，仅改渲染。控制参数见同目录 reproduction.txt。
"""
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';old=f.read_text()
if doc.splitlines()[0] not in old:f.write_text(old.rstrip()+'\n\n'+doc)
print(doc)
