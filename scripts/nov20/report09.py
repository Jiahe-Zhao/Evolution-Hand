
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from stack09 import out,OUT,np,E,R,json
from pathlib import Path
def result(k):return json.loads((out/(k+'.json')).read_text())
x=np.load(out/'custom.npy');r=result('custom');root=_evo_parse_xml(out/'stack09.xml');mass=[1000*sum(float(g.get('mass')) for g in root.findall(f".//body[@name='{n}']/geom")) for n in ('upper','lower')]
last=np.where(x[:,16]>0)[0][-1];land=np.where((x[:,17]>0)&(x[:,18]>0))[0][0];ang=np.rad2deg((R.from_quat(x[-240,3:7]).inv()*R.from_quat(x[-240:,3:7])).magnitude()).max()
assert r['success'] and land<last and r['lower_max_mm']<=10 and ang<1 and (x[-240:,19]==1).all() and result('face')['stable_s']>=1 and result('miss')['stable_s']==0 and result('edge')['stable_s']==0 and not result('idle')['success']
t=f"""## 09 不规则石料定向堆放：模型与验收

上石是不对称七边轮廓的凸多面体，约 20 × 12 × 9 mm、{mass[0]:.2f} g；侧面有斜肩，底面为指定稳定面。底石由四层不等宽八边轮廓构成，约 40 × 28 × 23 mm、{mass[1]:.2f} g；承托面小于最宽截面，底面着地。两块石均为六自由度刚体；凸包质量按密度 2500 kg/m³ 估算。稳定面内侧附薄的共面接触片，几何不伸出凸包，用于改善平面碰撞。石间和地面接触使用库仑摩擦（滑动系数 0.9），无吸附、焊接或落座锁定。系数是仿真参数，未做实物标定。

标准：手将上石放到指定底石，完全脱手后稳定 ≥1 s；底石全程相对初态三维位移 ≤10 mm。逐帧排除手接触任一石、上石接触地面或取料台；核对上下石实际接触、指定面法向偏差 ≤10°、相对中心高度 14±1.5 mm、水平偏心 <11 mm。最后 1 s 上石相对窗口起点平移 ≤1 mm，并复核角漂移 <1°。只约束稳定面方向，不限定面内偏航。

固定初态，hand_111 左手，Newton SolverMuJoCo CPU，240 Hz，仅驱动手部关节。拇指和食指同时接触时最大抬升 **{r['max_lift_mm']:.2f} mm**；{land/240:.3f} s 双指仍捏持时上下石已接触；{last/240:.3f} s 为最后一次手接触。到回合末连续脱手稳定 **{r['stable_s']:.3f} s**，底石全程最大三维位移 **{r['lower_max_mm']:.2f} mm**。末帧指定面法向偏差 {r['tilt_deg']:.4f}°；最后 1 s 上石最大平移 {r['final_upper_motion_mm']:.4f} mm、角漂移 {ang:.4f}°。**通过**；不是随机初态策略成功率。

物体对照：无手操作、正面居中释放后满足堆放终态 {result('face')['stable_s']:.3f} s；立边释放、偏离底石释放和原位空动作的合格保持时间均为 0 s。对照不计为手操作成功。

视频、两石逐步位姿、接触对、关节目标与结果：outputs/nov20/stack09/（custom.mp4、custom.npy、custom_contacts.json、custom_actions.npy、custom.json）。重放：IsaacLab3 Python 运行 scripts/nov20/stack09.py custom --video；复核：scripts/nov20/report09.py。
"""
p=OUT.parent.parent/'docs/SCENES_20_20261116.md';s=p.read_text();header=t.splitlines()[0]
if header not in s:p.write_text(s.rstrip()+'\n\n'+t)
(out/'RESULTS.md').write_text(t);print(t)
