
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from evaluate_long import stats
import csv,collections
o=OUT/'pen140';v=stats(o,'write');assert v[0]>=90 and v[1]<=3 and v[2]
a=np.load(o/'write.npz')['a'];h=np.load(o/'hover.npz')['a'];hh=(h[:,1]>0);gap=(h[:,6].min()-np.loadtxt(o/'hover_origin.txt')[3]-.001)*1000
assert h[:,9].max()==0 and gap>0 and np.linalg.norm(h[-1,4:6]-h[-1,2:4])<=.003 and (h[hh,10:12]>.01).all()
val=collections.defaultdict(lambda:np.zeros(len(a)));dist=collections.defaultdict(list);depth=collections.defaultdict(list)
for r in csv.DictReader(open(o/'write_rear.csv')):
 t=round(float(r['time_s'])*960);b=r['body'];force=float(r['force_N']);seg=r['rod_segment']
 if force<=.001:continue
 if a[t,1]>0 and b.startswith('link_'):dist[b].append(float(r['from_nib_mm']))
 if seg=='rear' and b.startswith('link_'):val[b][t]+=force;depth[b].append(float(r['gap_m']))
rows=[]
for k,g in sorted(val.items()):
 rows.append(f'|{k}|{100*(g>.001).mean():.1f}%|{g.max():.4f}|{-min(depth[k])*1000:.3f}|')
doc=f'''## 14 持笔描线：140 mm 长笔（2026-11-17）

保留原短笔验收。本版本长 140 mm、杆径 9 mm，从笔尖轴向量到名义握点 30 mm，握点后 110 mm。笔为一个自由刚体，笔杆采用单个连续圆柱碰撞体，消除握点处拼缝；另建过渡段和 1 mm 球形笔尖。拇指末节在书写段的实际接触点轴向位置中位数 {np.median(dist["link_1_3"]):.1f} mm，食指末节 {np.median(dist["link_2_3"]):.1f} mm，取点按 >1 mN 接触点样本统计。标称握点不强制锁定手指。

hand_111 左手，Newton SolverMuJoCo CPU，960 Hz，只驱动手，不对笔施力或设置轨迹。固定初态从已握笔开始，不计桌面拾笔。沿 52.95 mm 正弦曲线描线：笔尖与纸面真实接触且法向力 >0.01 N 才落墨；拇指、食指同时承载，只有相邻两步均合格才累计等弧长投影区间，重叠只计一次。**覆盖 {v[0]:.2f}%（要求 ≥90%），横向 RMS {v[1]:.3f} mm（要求 ≤3 mm）**。书写落墨时笔尖法向力中位数 {np.median(a[a[:,9]>.01,9]):.3f} N。独立悬空试次使用同规格笔、另一组初始倾角和手姿，平面目标曲线形状、尺寸相同；最小净空 {gap:.2f} mm，笔尖纸面接触力全程零，零墨点。

没有独立软组织虎口模型，以掌部、拇指/食指根部的刚体接触作为代理。下表统计整次正例，峰值先按物理步对该部件求和。负间隙是求解器接触层的少量重叠，不是生理压缩量。

|后杆接触部件|时间占比|峰值 N|最大负间隙 mm|
|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

复现（scripts/nov20，IsaacLab3 Python）：依次设置 AX=.18 AY=-.4 运行 long14.py write t4、再加 TH=-.025 运行 monorod.py write mono1 和 final_long.py write write；AX=.7 AY=-.4 运行 final_long.py hover hover（不加 TH）。前两次用于准备已握笔的初态。最后运行 render_long.py。

视频左侧纸面显示目标线和实际落墨，右侧显示同步俯视轨迹。视频、轨迹叠图、逐步笔尖与握持力、逐接触点记录和场景文件在 outputs/nov20/pen140/。固定初态结果不代表随机初态成功率。
'''
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';old=f.read_text();assert doc.splitlines()[0] not in old;f.write_text(old.rstrip()+'\n\n'+doc);print(doc)
