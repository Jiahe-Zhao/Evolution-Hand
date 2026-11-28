
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from bead10 import *
from scipy.spatial.transform import Slerp
tag='wall4_0';x=np.load(out/(tag+'.npy'));aligned=np.load(out/'aligned.npy');skew=np.load(out/'guide35.npy');c=json.loads((out/'guide35_contacts.json').read_text())
# Verify the interpolated replay in addition to actual Newton states. Far from
# rod, bounding sphere separates outer bead from finite cylinder unequivocally.
mins=[];times=np.arange(len(x));interp=Slerp(times,R.from_quat(x[:,3:7]));ts=np.linspace(0,len(x)-1,(len(x)-1)*8+1);qs=interp(ts).as_quat();ps=np.stack([np.interp(ts,times,x[:,i]) for i in range(3)],axis=1)
for pos,q in zip(ps,qs):
 if np.linalg.norm(pos[:2]-xy)>np.hypot(.010,H)+rad:continue
 pose=np.r_[pos,q];rot=R.from_quat(q).inv();d=rot.apply([0,0,1]);tip=rot.apply(np.r_[xy,zt]-pos);foot=tip-(zt-zb)*d;extent=rad*np.linalg.norm(d[:2])
 if max(tip[2],foot[2])+extent < -H or min(tip[2],foot[2])-extent>H:continue
 depth,gap,angle=measure(pose);mins.append(gap)
min_gap=min(mins);held=x[:,10]>0;idx=np.flatnonzero(held&(x[:,7]>=10)&(x[:,8]>0)&(times>525))[0];lift=(x[held,2].max()-up[2])*1000;skdrift=1000*np.linalg.norm(skew[-120:,:3]-skew[-120,:3],axis=1).max();assert np.load(out/'guide0.npy')[-1,7]>=10 and min_gap>0 and lift>10 and x[:,12].max()==0 and x[-1,10] and x[-1,7]>=10 and aligned[-1,7]>=10 and skew[-1,7]<10 and skew[-120:,13].all() and not skew[-120:,14].any() and skdrift<.5
t=f"""## 10 骨珠穿杆：贯通孔与偏斜接触

自由骨珠外廓约 20 × 14.4 × 12 mm，椭圆鼓形外壁；12 个独立凸扇段组成环，每段同时构成可碰撞内壁，中心不填充。孔正十二边形，最窄对边距 {2000*hole*np.cos(np.pi/N):.3f} mm；固定圆杆直径 {2000*rad:.2f} mm、露出 17 mm。最窄单边径向间隙 {(hole*np.cos(np.pi/N)-rad)*1000:.3f} mm。孔和杆均无吸附、位姿跟随或穿入触发器。骨珠质量根据分块体积及 1900 kg/m³ 估算；摩擦为仿真参数。

验收：手实际双指捏起并从杆外把孔送到杆尖上方；按孔轴计算杆尖距入口端面的距离（上限为有效孔长 12 mm），手持时至少 10 mm。深度只在杆确实位于孔内时有效，落在杆旁不能得分。用每个孔内壁面的半空间检查杆的整个斜截面，而非只看杆轴心；远离时用外包球排除碰撞。接触、位姿均来自同一次 Newton SolverMuJoCo CPU 仿真，240 Hz。

|试次|现象及数值|
|---|---|
|正向重力释放|无手接触，对正滑入，最终 {aligned[-1,7]:.2f} mm。|
|偏斜受阻探针，35° 目标倾角|最终孔轴倾角 {skew[-1,9]:.2f}°，杆尖轴向进入约 {skew[-1,7]:.2f} mm；末段 0.5 s 连续珠杆接触、不碰底座，珠心最大漂移 {skdrift:.3f} mm。未达到 10 mm。此探针按 960 Hz 步进，仅施加姿态恢复力矩模拟操作者保持偏斜，平动自由，无轴向推力。该试次求解器峰值接触压入 {skew[:,12].max():.3f} mm，非理想刚体。相同姿态夹具 0° 时能滑入 12 mm。另测完全自由的 35° 重力释放：1.5 s 时只进入约 4.45 mm，延长到 6 s 后自行摆正并滑入；不能据此声称自由珠必然永久卡死。|
|hand_111 左手，标准槽外取料台初态|仅写手指和腕关节目标。拇指、食指末节同时捏持时抬升 {lift:.2f} mm；{idx/240:.3f} s 双指持珠越过 10 mm；结束时仍捏持，套入 {x[-1,7]:.2f} mm。通过。|

防侧穿复核：正例全程 Newton 未记录珠壁与杆的负接触距离；对原始位姿及帧间 8 等分的线性位置 / 球面旋转插值，进入孔内时最小保守壁间隙 {min_gap:.3f} mm。未声称连续时间碰撞检测或随机初态成功率。

脚本 scripts/nov20/bead10.py；IsaacLab3 Python 在 Evolution_PC 根目录重放：python scripts/nov20/bead10.py {tag} --video。数据和视频位于 outputs/nov20/bead10/：{tag}_actions.npy、{tag}.npy、{tag}_contacts.json、{tag}.mp4、aligned.mp4、guide35.mp4、guide0.mp4、skew35.mp4。"""
t=t.replace('标准槽外取料台初态','杆外取料台初态');p=OUT.parent.parent/'docs/SCENES_20_20261116.md';s=p.read_text();hdr=t.splitlines()[0]
if hdr not in s:p.write_text(s.rstrip()+'\n\n'+t+'\n')
(out/'RESULTS.md').write_text(t+'\n');print(t)
