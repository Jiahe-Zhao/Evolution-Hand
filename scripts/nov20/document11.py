
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json
r=_EVOLUTION_ROOT;p=r/'docs/SCENES_20_20261116.md';out=r/'outputs/nov20/inlay11'
rows=[]
for mode,label in [('aligned','正向自由落槽'),('reverse','反向自由落槽'),('rim','搭边'),('hand','当前手关节轨迹')]:
 x=json.loads((out/(mode+'.json')).read_text());rows.append(f"|{label}|{x['angle_deg']:.2f}|{x['depth_mm']:.2f}|{x['seated_s']:.3f}|")
section="""
## 11 细石叶定向嵌槽：模型与验收

固定木柄，顶部开槽；自由刚体为无锐刃、单角截短的五边形石叶代理，包络 20 × 12 × 8 mm。槽底和四侧主体使用实体盒，截角定位块限制反向装入；槽深 6 mm。配合依靠重力和碰撞，不吸附、不锁定。验证的是顶部嵌放，不能据此外推倒置或抗拔能力。

手操作验收：从槽外起始（初始偏航差约 29°），拇指、食指末节同时接触并抬升至少 10 mm；持物调向后，方向误差 ≤10°，石叶底面最浅角进入槽口 ≥4.8 mm，中心水平偏差 <2 mm；完全松手后连续保持至少 0.5 s 至回合结束。反向、斜搭、只推入、手仍支撑均不算通过。

Newton SolverMuJoCo CPU，240 Hz。独立模型探针只改变初态，之后自由下落。深度为末帧有符号深度，负值表示有底角在槽口上方。表中保持时间为最长连续满足终局几何且无手接触的时段。

|试次|末帧方向误差 °|末帧深度 mm|保持 s|
|--|--:|--:|--:|
"""+'\n'.join(rows)+f"\n\n手部完整正例：{'通过' if x['success'] else '未通过'}。脚本：scripts/nov20/inlay11.py；视频、逐步位姿和汇总：outputs/nov20/inlay11/。\n"
s=p.read_text().split('\n## 11 细石叶定向嵌槽：模型与验收')[0];p.write_text(s.rstrip()+'\n'+section);(out/'RESULTS.md').write_text(section)
