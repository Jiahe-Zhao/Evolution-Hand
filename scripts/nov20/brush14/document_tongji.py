
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,sys,hashlib
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1]
m=json.loads((P/(tag+'_writing_metrics.json')).read_text());comp=json.loads((P/(tag+'_comparison.json')).read_text())
rows=['# hand_111 毛笔书写同济：物理控制优化（2026-11-22）','','AX50 适配继续暂停。本次沿用 hand_111、自由笔杆和原 13 束 / 65 段被动毛束，从预置执笔姿态开始；只控制手指和手腕，无笔杆锁定、外加笔杆力或墨迹轨迹替换。','','## 控制修改','','- 接触中心 40 ms、法向力 50 ms 低通滤波，仅用于反馈；验收仍使用原始物理接触记录。','- 手腕 XY 纠偏速度限制 3 mm/s、加速度限制 20 mm/s²；Z 力反馈速度限制 1.5 mm/s。','- 各目标折线段使用五次平滑速度曲线，转角减速；不修改原始目标字形。','- 中指、无名指保留几何接触修正，目标关节每 50 ms 最大变化 0.008 rad；加入初始握姿约束，避免累计漂移。','- 第一轮冻结握姿虽降低晃动，但丢失中指和无名指支撑，已拒绝；保留 tongji_smooth_v1 原始结果。','','## 不变的验收标准','','逐笔覆盖率 ≥90%，接触中心横向 RMS ≤3 mm；有效接触采样四指均 >0.005 N；悬空不落墨；求解器无失败与警告。','本次总验收：'+('通过' if m['success'] else '未通过')+'。不以视觉顺滑替代精度与接触要求。','','## 前后对比','','|指标|原版|本次|','|---|---:|---:|']
a,b=comp.values()
for key,label in [('tilt_peak_deg','最大笔杆倾角（度）'),('writing_tilt_rms_deg','书写时倾角 RMS（度）'),('brush_position_above_5Hz_rms_mm','实际笔杆位置 >5 Hz RMS（mm）'),('brush_tilt_above_5Hz_rms_deg','实际笔杆倾角 >5 Hz RMS（度）'),('weighted_coverage_pct','弧长加权覆盖率（%）'),('max_stroke_rms_mm','最差笔画 RMS（mm）'),('passed_strokes','通过笔画数 /15'),('air_contact_count','悬空墨点数')]:rows.append(f'|{label}|{a[key]:.6g}|{b[key]:.6g}|')
rows+=['','高频波动使用 480 Hz 原始物理状态、三阶 5 Hz 高通，统计书写阶段 RMS；不是渲染结果，亦不等价于全部可见曲折。时间曲线按各自仿真时间展示，新旧动作时长不同。','','## 逐笔结果','','|笔画|覆盖率 %|接触中心 RMS mm|四指接触比例|通过|','|---|---:|---:|---:|---|']
for s in m['strokes']:rows.append(f"|{s['stroke']}|{s['coverage_pct']:.2f}|{s['contact_centroid_rms_mm']:.3f}|{s['four_finger_contact_fraction']:.5f}|{'是' if s['pass'] else '否'}|")
rows+=['','## 证据','',f'[三视角视频](../outputs/nov20/brush14/{tag}_dense_brush_views.mp4)、[实际轨迹叠图](../outputs/nov20/brush14/{tag}_overlay.png)、[前后曲线](../outputs/nov20/brush14/{tag}_comparison.png)、[验收 JSON](../outputs/nov20/brush14/{tag}_writing_metrics.json)、[原始接触](../outputs/nov20/brush14/{tag}_contacts.csv)、[控制记录](../outputs/nov20/brush14/{tag}_controller.csv)。','','视频沿用原三视角、900 根显示毛丝及接触墨点去重方式，2 倍速；未额外平滑显示墨迹。','',f'模型 XML SHA256：{hashlib.sha256((P/(tag+".xml")).read_bytes()).hexdigest()}。原版模型是否完全一致：{(P/(tag+".xml")).read_bytes()==(P/"tongji_hand111.xml").read_bytes()}。']
doc=P.parents[2]/'docs/TASK14_HAND111_TONGJI_SMOOTH.md';doc.write_text('\n'.join(rows)+'\n');print(doc)
