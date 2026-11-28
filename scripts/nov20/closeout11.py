
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np,json,sys
r=_EVOLUTION_ROOT;o=r/'outputs/nov20/inlay11';mode=sys.argv[1];x=np.load(o/(mode+'.npy'));j=json.loads((o/(mode+'.json')).read_text());dual=x[:,9].astype(bool);h=dual&(x[:,2]>.0414);seat=x[:,11].astype(bool)
end=int(len(seat)-1-np.flatnonzero(~seat)[-1]) if (~seat).any() else len(seat)
text=f"""### 11 下降段接触复核与连续操作验收
原 place 轨迹逐步复现（位姿、评分列最大差 0）。240 Hz，从第 455 步开始查求解器接触对（dist ≤ 0），未检测到手与木柄槽沿接触。最后双指捏持在第 596 步；第 597 步起失去双指捏持，原定下降起点是第 600 步，第 618 步石叶首次碰前槽沿。原失败直接表现是指间失稳，未发现槽沿先挡住指尖的证据；保留物体及原判据。

最新实测 {mode}：手型 hand_111 左手；Newton SolverMuJoCo CPU，240 Hz，从槽外原始初态开始，只写手的关节目标。双末节同时接触时最大抬升 {(x[dual,2].max()-.0314)*1000:.2f} mm；抬起后持物最小方向误差 {x[h,7].min():.2f}°。末帧方向误差 {x[-1,7]:.2f}°，深度 {x[-1,8]:.2f} mm，水平中心误差 {1000*np.linalg.norm(x[-1,:2]-[-.102,.030]):.2f} mm，回合末连续无手接触且合格保持 {end}/240 = {end/240:.3f} s。完整判定：{'通过' if j['success'] else '未通过'}。固定初态轨迹验证，非策略成功率。

证据：outputs/nov20/inlay11/{mode}.mp4、{mode}.json、{mode}.npy、{mode}_actions.npy、{mode}_contacts.json；原失败复核 audit_contacts.json / audit_states.npy。
"""
for p in [r/'docs/SCENES_20_20261116.md',o/'RESULTS.md']:
 s=p.read_text().split('### 11 下降段接触复核与连续操作验收')[0];p.write_text(s.rstrip()+'\n\n'+text)
print(text)
