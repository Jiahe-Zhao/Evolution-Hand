
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from grind12 import OUT,out,np,mj,json
from score12 import assess
from pathlib import Path
f=np.load(out/'verified.npz');x,ok,ts,dist,turn=assess(f);metrics=json.loads((out/'metrics.json').read_text());assert metrics['tag']=='verified' and len(dist)==6 and min(dist)>=30 and metrics['hold_fraction']==1 and not any(metrics['one_end_samples']) and np.all(turn[:,3]==0)
assert np.array_equal(x,np.load(out/'soft5.npz')['data']) if (out/'soft5.npz').exists() else True
m=mj.MjModel.from_xml_path(str(out/'verified.xml'));mass=m.body_mass[m.body('stone').id];t0,t1=ts[0],ts[-1];w=x[t0:t1+1];window=np.concatenate([np.arange(k-48,k+49) for k in ts[1:-1]]);assert ok[window].all()
np.savetxt(out/'force_trace.csv',np.column_stack([x[:,0],x[:,1]*1000,x[:,8:14],ok.astype(int)]),delimiter=',',header='time_s,stone_x_mm,normal_N,front_N,rear_N,contact_hull_mm2,tilt_deg,opposed_grip,eligible',comments='',fmt='%.7g')
text=f"""## 12 往复研磨：自由磨石、浅凹磨盘与换向验收

固定长槽形浅凹盘，沿行程方向平直，横向五段拼成连续工作面（184 × 90 mm，中心至边缘高差 3.5 mm）。自由磨石约 {1000*mass:.1f} g，工作面最长 70 mm、宽 42 mm，用三块刚性平面拼合，中央平面和两侧斜面与磨盘横向轮廓相配；握部两端加宽以承受交替的推拉。仅夹持与接触传力，无焊接、滑轨或外加磨石驱动。此场景验证面研磨运动，不模拟粉末和磨损。

hand_111 左手，Newton SolverMuJoCo CPU，960 Hz。每步从求解器读取磨石工作面与盘面各接触点的法向力并相加；不含手压磨盘、手握磨石的力。仅向手部关节给目标，用测得的盘面力反馈调节腕高。以磨石（非手腕）的实际折返点分段。每个步间隔的两端都需满足：拇指与至少一个长指有载荷接触、总法向力 0.5–5 N、前后半面各承载 >0.01 N，且负载点投影凸包 ≥10 mm²。凸包仅代表支撑范围，并非微观真实接触面积。仅累加沿该程方向新达到的位置增量，往回抖动及补走旧位置不重复计分；无载、单边、超限均排除。

|周期|去程有效 / 实际 mm|回程有效 / 实际 mm|
|---|---:|---:|
"""
for i in range(3):
 j=i*2;text+=f"|{i+1}|{dist[j]:.2f} / {metrics['actual_mm'][j]:.2f}|{dist[j+1]:.2f} / {metrics['actual_mm'][j+1]:.2f}|\n"
text+=f"""
三周期六个单程均通过。有效样本力范围 {metrics['effective_N'][0]:.3f}–{metrics['effective_N'][1]:.3f} N；含不计分段的原始范围 {metrics['raw_N'][0]:.3f}–{metrics['raw_N'][1]:.3f} N，有效样本占 {metrics['eligible_fraction']*100:.1f}%。六程内未记录承载的时间累计 {metrics['zero_load_s']:.4f} s，不计程。全段保持对指握持。

以五个内部换向点各 ±50 ms 审计：五个窗口均全程力合格且前后半面有载；最大倾角 {turn[:,0].max():.3f}°，中央工作面前后端最大垂直高度差 {max(metrics['edge_delta_mm']):.3f} mm，任一半面最低仍承载 {turn[:,1].min():.3f} N，未出现单边翘起或整体离盘。不能由此推断换向外的全部行程均无卸载。

复跑：在 Evolution_PC 根目录，用 IsaacLab3 Python 依次执行 scripts/nov20/grind12.py verified、scripts/nov20/deliver12.py verified（MUJOCO_GL=egl）、scripts/nov20/report12.py。outputs/nov20/grind12/ 下保存 verified.xml、verified.npz（逐步力、位姿与关节控制）、metrics.json、force_trace.csv、force_curve.png 和 operation.mp4；视频按同一轨迹 30 fps 渲染，不用于采样力峰。仅固定初态正例。
"""
p=OUT.parent.parent/'docs/SCENES_20_20261116.md';s=p.read_text()
if text.splitlines()[0] not in s:p.write_text(s.rstrip()+'\n\n'+text)
(out/'RESULTS.md').write_text(text);print(text)
