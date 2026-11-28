
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
from PIL import Image,ImageFont,ImageDraw
import numpy as np,json,subprocess as sp, shutil,hashlib
from build import OUT, ROOT
dest=OUT/'review';dest.mkdir(exist_ok=True)
tasks=['carry','stone','handle','phone'];titles=['03  掌中搬运','06  核心石掌内换面','13  活动提梁','18  单手持机点选']
fontpath=str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc')
font=ImageFont.truetype(fontpath,21,index=2);small=ImageFont.truetype(fontpath,16,index=2)
videos=[];metrics=[];rows=[]
for t in tasks:
 for part in ['_gpu','_negative_gpu','_gpu_contrast']:
    for ext in ['json','npz']+(['mp4'] if part!='_gpu_contrast' else []):
     p=OUT/(t+part+'.'+ext);shutil.copy2(p,dest/p.name)
 rec=[json.loads((dest/(t+k+'.json')).read_text()) for k in ['_gpu','_negative_gpu','_gpu_contrast']]
 assert [r['success'] for r in rec]==[True,False,False],rec
 x=np.load(dest/(t+'_gpu.npz'))['metrics'];metrics.append(x);rows.append(rec)
 buf=sp.check_output(['ffmpeg','-loglevel','error','-i',str(dest/(t+'_gpu.mp4')),'-f','rawvideo','-pix_fmt','rgb24','-'])
 videos.append(np.frombuffer(buf,dtype='u1').reshape(-1,360,480,3))
total=max(map(len,videos))
proc=sp.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','960x828','-framerate','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'four_tasks.mp4')],stdin=sp.PIPE)
for i in range(total):
 canvas=Image.new('RGB',(960,828),'#132431');draw=ImageDraw.Draw(canvas)
 for j,frames in enumerate(videos):
    x0=j%2*480;y0=j//2*414;idx=min(i,len(frames)-1);m=metrics[j][idx*8];canvas.paste(Image.fromarray(frames[idx]),(x0,y0+31))
    draw.text((x0+10,y0),titles[j],font=font,fill='white')
    draw.text((x0+385,y0+4),('结束' if i>=len(frames) else f'{m[0]:.2f} s'),font=small,fill='#a7c8cb')
    line=[f'水平目标误差 {m[2]*1000:.1f} mm   掌内漂移 {m[1]*1000:.1f} mm',f'目标面误差 {m[3]:.1f}°   掌内漂移 {m[1]*1000:.1f} mm',f'提梁 {np.rad2deg(m[4]):.0f}°   器身抬升 {m[5]*1000:.1f} mm',f'{["等待第一点","第一点命中","已离屏","第二点命中"][int(m[7])]}   机身滑移 {m[1]*1000:.1f} mm'][j]
    draw.text((x0+10,y0+390),line,font=small,fill='#ddedf0')
 proc.stdin.write(canvas.tobytes())
 if i==87:canvas.save(dest/'four_tasks.jpg',quality=92)
proc.stdin.close();assert proc.wait()==0
m0,m1,m2,m3=[x[-1] for x in metrics];phone=rows[3][0]
report=f"""# 2026-11-16：Newton 四场景物理验收
固定初态，hand_111 左手，RTX 4090，Isaac Lab 3.0.0 对应 Python 环境，直接 Newton SolverMuJoCo / MJWarp CUDA。每任务三次回放：关节控制正例、无动作反例、针对性反例。不是三个随机种子；不估计策略成功率。视频为实际仿真状态重绘，无物体状态覆盖。15 项新增任务本批实现 3 项，carry 为已有场景迁移。物体为刚体代理，携带食物为椭球。手由腕部机构驱动，提梁组起始时指尖已靠近横梁，手机从托持态开始。

|任务|物理正例|全程空动作|针对性反例|
|---|---|---|---|
|搬运|终点水平误差 {m0[2]*1000:.2f} mm，掌内漂移 {m0[1]*1000:.2f} mm，连续达标 {rows[0][0]['best_s']:.2f} s|未通过|仅搬一半距离，未通过|
|掌内换面|相对掌目标面误差 {m1[3]:.2f}°，石心漂移 {m1[1]*1000:.2f} mm，连续达标 {rows[1][0]['best_s']:.2f} s|未通过|仅转腕，未通过|
|活动提梁|器身抬升 {m2[5]*1000:.2f} mm，提梁 {np.rad2deg(m2[4]):.1f}°，器身倾斜 {m2[6]:.1f}°，连续达标 {rows[2][0]['best_s']:.2f} s，无手直接触器身|未通过|抬腕行程不足，未通过|
|持机点选|首次命中 {phone['events'][0]['time']:.3f} s，连续离屏确认 {phone['events'][1]['time']:.3f} s，第二点命中 {phone['events'][2]['time']:.3f} s，全程最大掌内滑移 {metrics[3][:,1].max()*1000:.3f} mm（阈值 12 mm）|未通过|仅点第一点，未通过|

四个正例和八个反例，全部符合预期。每条轨迹 {1/rows[0][0]['dt']:.0f} Hz 记录关节目标及判据；视频 30 fps。四宫格 5 秒，较短任务结束后停在其末帧。

复跑见 scripts/nov20/README.md；场景标准见 docs/SCENES_20_20261116.md。输出位于 outputs/nov20/review：每项 JSON 标明通过与否，NPZ 保存轨迹，MP4 为预览。尚未验证随机初态、另一种手型及并行训练；手机滑移余量小，正式评测前优先检查。
"""
(dest/'RESULTS.md').write_text(report);shutil.copy2(ROOT/'docs/SCENES_20_20261116.md',dest/'SCENES.md')
files=[ROOT/'scripts/nov20'/k for k in ('env.py','run_final.py','build.py','gym_tasks.py')]+[OUT/(t+'.xml') for t in tasks]
(dest/'sources.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
print(report)
