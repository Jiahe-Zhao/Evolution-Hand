
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,json
r=_EVOLUTION_ROOT;o=r/'outputs/nov20/latch16'
def get(k):return np.load(o/(k+'.npy')),json.loads((o/(k+'_contacts.json')).read_text())
x,e=get('hand');d,de=get('direct');l,le=get('locked');c,_=get('clear')
u=np.flatnonzero(x[:,1]>=8)[0];t=np.flatnonzero(x[:,2]>=60)[0]
tip=next(i for i,cs in enumerate(e) if any('bolt/tab|link_2_3' in a for a in cs));pull=any('lid/pullbar|link_' in a for cs in e[u:t+1] for a in cs)
assert tip<u<t and max(x[:u,2])<3 and pull and x[-1,2]>=60
assert max(d[:,2])<3 and max(l[:,2])<3 and max(c[:,2])>=60 and max(abs(d[:,1]))<1 and any('bolt/tongue|lid/ear' in cs for cs in de)
txt=f"""## 16 拨闩开盖：盒盖与滑闩物理验收

固定中空盒身（底板、端壁、侧壁），被动铰链盖、外露拉环、盖耳及独立被动直线滑闩。盖板约 44 × 50 × 4 mm；锁舌初始与盖耳侧向重叠 8 mm，退闩行程 18 mm。通过刚体接触阻止盖耳上行；关节导轨约束滑闩，既不切换碰撞，也不按任务阶段锁定或解锁铰链。无开盖弹簧，试次末手仍托盖。

原验收：锁住时直接拉不开，先解除锁舌重叠，再开盖 ≥60°。测试中以退闩 ≥8 mm 为几何解锁，直接拉时盖角 <3° 为闭锁对照容差。Newton SolverMuJoCo CPU，240 Hz，hand_111 左手，标准初态。正例和直接拉的反例均只给手部关节目标，未驱动盖或闩。

|同一模型的试次|实测|
|---|---|
|锁住，跳过拨闩，直接用手拉环|盖耳接触锁舌；锁闩最大位移 {max(d[:,1]):.2f} mm，最大开角 {max(d[:,2]):.2f}°|
|机构力矩对照，锁住 / 初始预退闩 18 mm|同施 0.025 N·m 开盖力矩（前 30 步线性加载，共 240 步），最大开角 {max(l[:,2]):.2f}° / {max(c[:,2]):.2f}°；后者末帧 {c[-1,2]:.2f}°。铰链上限为软限位，加载时有过冲，此对照只检验闭锁及解锁后可动性|
|手先拨闩再开盖|食指末节接触拨片，{u/240:.3f} s 退过 8 mm；随后手经拉环打开盒盖，{t/240:.3f} s 首次达到 60°；末帧 {x[-1,2]:.2f}°，滑闩 {x[-1,1]:.2f} mm。通过|

视频（左：完整场景；右：同帧隐藏手，显示机构）、逐步关节状态及接触对：outputs/nov20/latch16/hand.mp4、direct.mp4、各模式 .npy 和 *_contacts.json。源文件 scripts/nov20/latch16.py；用 IsaacLab3 的 Python 分别运行 locked / clear / direct / hand；scripts/nov20/report16.py 复核。固定初态验证，未测随机初态成功率。
"""
p=r/'docs/SCENES_20_20261116.md';s=p.read_text();header='## 16 拨闩开盖：盒盖与滑闩物理验收';assert header not in s;p.write_text(s.rstrip()+'\n\n'+txt);(o/'RESULTS.md').write_text(txt);print(u,t,x[-1])
