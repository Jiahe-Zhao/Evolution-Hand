
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from stack09_multi import out,OUT,np,R,json
def read(mode):return np.load(out/(mode+'.npy')),json.loads((out/(mode+'.json')).read_text())
x,a=read('adjusted');f,b=read('custom');w,c=read('testpulse0.10');s,d=read('testpulse0.55')
rot=R.from_quat(x[-240:,3:7]);angle=np.rad2deg((rot[0].inv()*rot).magnitude()).max()
assert angle<1 and a['success'] and not b['success'] and c['success'] and not d['success'] and np.all(x[-240:,20]==3) and not x[-240:,21].any()
assert np.allclose(x,w[:len(x)]) and np.allclose(x,s[:len(x)])
t=f"""## 09B 五石不对称支撑堆放
09 的独立变体，不改原双石场景、不增加月度场景计数。三块自由支脚石呈不对称三角布局，上放原承托石，再由 hand_111 左手从取料台放上第五块石头。支脚初始中心相对承托石的水平偏移为 (-12,-6)、(12,-5)、(-3,9) mm；三个小承托面高差 0.50 mm。五块均为带重力、摩擦和碰撞的自由刚体，无固定或黏结。支脚和承托石预置，手只放最后一块。评价底石位移时沿用 09 的承托石。视频右侧隐藏手，仅用于观察接触。

同一 Newton 模型、初态、240 Hz；两条动作前 710 步相同。调整后的放置：双指最大抬升 {a['max_lift_mm']:.2f} mm；完全撤手后满足堆放条件至末帧 {a['stable_s']:.3f} s，承托石全程最大三维位移 {a['lower_max_mm']:.2f} mm。最后 1 s 的每一步均检测到三个支脚与承托石接触、承托石未触地、手未接触任何石块；上石最大平移 {a['final_upper_motion_mm']:.3f} mm、角漂移 {angle:.3f}°。通过 1 s / 10 mm 原标准。
原双石动作直接复用：不施加扰动亦失稳，合格尾段 {b['stable_s']:.3f} s，承托石最大位移 {b['lower_max_mm']:.2f} mm。两种结果说明该结构对放置和撤手动作敏感。

扰动对照从成功轨迹的末态开始，向承托石施加世界 +X 方向水平力 0.10 s。0.10 N 后仍保持堆放；0.55 N 后失稳。两档实测，不推断临界力或实际材料强度。未测随机初态成功率。

远端脚本 scripts/nov20/stack09_multi.py（adjusted / custom / testpulse0.10 / testpulse0.55，加 --video），数据 outputs/nov20/stack09_multi/：逐帧五石位姿、接触对、adjusted_actions.npy、各试次视频及 JSON。"""
p=OUT.parent.parent/'docs/SCENES_20_20261116.md';h=t.splitlines()[0];orig=p.read_text()
if h not in orig:p.write_text(orig.rstrip()+'\n\n'+t+'\n')
(out/'RESULTS.md').write_text(t+'\n')
