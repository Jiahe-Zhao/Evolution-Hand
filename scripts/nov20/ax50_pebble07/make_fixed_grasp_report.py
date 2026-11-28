
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json
r=Path((str(_EVOLUTION_ROOT)));o=r/'outputs/nov20/ax50_pebble07';tags=['vector_fixed_grasp_50g_1128','vector_fixed_grasp_v2_50g_1128'];a,b=[json.loads((o/(t+'_summary.json')).read_text()) for t in tags]
text='''# AX50 修复接触模型后 50 g 石子复验 · 2026-11-28

结论：本轮未通过稳定提起 2 mm。完成实际闭合、第一轮提拉、第二轮延后提拉条件对照。没有加大摩擦，没有把手腕位移算成石子位移。

## 条件与控制

沿用自由的圆滑、轻微上宽下窄石子，质量实测约 50 g；托盘和邻石保留。修复模型为等价端点接触去重、目标软接触罚刚度 1125 N/m、接触阻尼 0.5。软垫材料未变。有效软摩擦系数在运行中读取仍为 0.357071；刚体原生接触规则维持 0.9。

从原预接触状态检查点继续，保留状态与速度、清除旧软反馈，先稳定 0.2 s；求解器 warm-start 不恢复。这不是全任务从远处接近的重新验收。石子没有绑定、吸附或直接位姿驱动。仅控制手部。

双指分别通过 IK 沿水平方向相向闭合，指令速度 0.8 mm/s，每指最大指令 2 mm；目标平均法向力为拇指 0.65 N、食指 0.85 N。原手指关节力矩上限 0.02 Nm 保留。控制以实际反馈调节闭合，提拉期间仍可补偿。

原逐子步保护保留：2 N 瞬时负载或 0.2 mm 刚体重叠触发退让，关节目标限速 0.1 rad/s；另把目标软接触重叠纳入同一退让检查。0.8 mm 为原紧急停止阈值。这些是退让控制，不是保证零过冲的硬约束，不能因保护启用就宣称穿透合格。

第一轮启动判据：连续 0.2 s 两指 5 ms 平均力均超过 0.15 N，窗口力变异系数不超过 20%，石子窗口位移范围不超过 0.2 mm，速度低于 3 mm/s，无其他手部接触；达到后指令提拉 2 mm。该判据只证明短时双指接触，不能证明已具备承重能力。

第二轮保持同样摩擦、模型、力矩和保护，继续闭合，增加简化竖直承载估计持续超过 0.65 N 的启动条件（约为重力的 1.3 倍）。估计包含实测法向竖直分量、刚体切向摩擦容量及软接触弹性法向对应的竖直摩擦容量，不把软阻尼尖峰当摩擦容量；它不是实际承重或力矩平衡的证明。若触发则指令提拉 2.2 mm，以测量实际是否达到 2 mm。启动门槛不替代验收。

本轮 2 mm 判据：石子相对提拉起点上升至少 2 mm，托盘竖直支撑低于 0.01 N，两指保持接触、无其他手部支撑，连续保持 0.5 s。原完整 20 mm 任务验收未改。

## 结果

'''
text+='| 指标 | 第一轮 | 第二轮 |\n|---|---:|---:|\n'
for label,key in [('实际触发提拉','micro_lift_attempted'),('保持 2 mm 时长 s','stable_lift_hold_s'),('最大软接触重叠 mm','max_soft_overlap_mm'),('最大刚体重叠 mm','max_rigid_depth_mm'),('最大软垫法向形变 mm','max_soft_compression_mm'),('最大邻石位移 mm','max_neighbor_displacement_mm')]:
 vals=[]
 for x in [a,b]:
  v=x[key];vals.append(('是' if v else '否') if isinstance(v,bool) else f'{v:.6g}')
 text+='| '+label+' | '+' | '.join(vals)+' |\n'
text+=f"\n第一轮在 {a['trigger_s']:.3f} s 启动提拉；手腕最终上升 {a['final_wrist_rise_mm']:.3f} mm，石子最大高度变化 {a['max_lift_mm']:.3f} mm，结束相对提拉起点 {a['final_lift_mm']:.3f} mm。石子未稳定离开托盘。第二轮状态 `{b['stop']}`，提拉是否执行：{b['micro_lift_attempted']}。\n"
w=a['window_summary'];text+=f"\n第一轮提拉斜坡之后，拇指平均竖直作用力 {w['thumb_fz']['mean']:.5f} N；食指法向竖直分量 {w['index_soft_nz']['mean']:.5f} N，摩擦竖直分量 {w['index_soft_tz']['mean']:.5f} N，食指净竖直力 {w['index_soft_fz']['mean']:.5f} N；托盘平均向上支撑 {w['support_fz']['mean']:.5f} N。三者合计约 0.4908 N，与 0.4905 N 重力接近，说明仍由托盘承重且食指总体向下压。正 Z 为向上。\n"
text+=f"\n第一轮闭合指令最终为 {a['final_closure_mm']} mm，但拇指骨段参考点净向内位移 {a['thumb_actual_inward_final_mm']:.3f} mm（负值为向外），食指为 {a['index_actual_inward_final_mm']:.3f} mm。参考点运动包含腕部让位，不等于关节独立运动或软表面运动；未记录实际驱动力矩饱和和逐次退让计数，不能据此唯一归因于驱动限制或保护。\n"
text+=f"\n第二轮末段简化承载估计均值 {b.get('potential_up_final_window_mean_N',0):.4f} N，窗口最小值 {b.get('potential_up_final_window_min_N',0):.4f} N；未将未触发的提拉写成成功。\n"
text+=('\n## 压陷、穿透与剩余问题\n\n软垫法向形变是接触点相对骨段随动的静止形状，在接触法线方向上的表面位移；包含压弯和拖动，不等于皮肤厚度压缩率。软接触重叠和原生刚体重叠另行记录，三者不得混用。\n\n第一轮刚体重叠超过 0.2 mm 退让阈值，存在短时过冲，不能标为穿透验收通过。自由石子情况下仍有明显子步软反力峰；平面标定改善不能替代动态软硬耦合验证。\n\n现有证据支持：短时双指接触没有转化为稳定承重；提拉期间食指净力向下，拇指支撑点未稳住。尚不能仅凭本轮区分接触位置、腕部/手指让位、控制保护与动态软硬耦合各自贡献。下一步应优先检查拇指支撑稳定性和食指接触法线方向，保持摩擦，避免直接增加闭合量或摩擦掩盖问题。\n\n## 文件\n\n原始仿真与数据在 `' + str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_pebble07/`，前缀为 `vector_fixed_grasp_50g_1128`、`vector_fixed_grasp_v2_50g_1128`，含 `_trace.csv`、`.npz`、`_friction.npz`、`_result.json`、`_summary.json`、`_initialization.json`、`_mass_audit.json`。\n\n视频为保存的实际物理状态，30 Hz 取帧、7.5 fps 编码，4 倍慢放，无轨迹插值。画面显示 5 ms 平均指力，子步峰值保留在数据与曲线中。\n\n![第一轮曲线](../outputs/nov20/ax50_pebble07/vector_fixed_grasp_50g_1128_curves.png)\n\n![三维力](../outputs/nov20/ax50_pebble07/vector_fixed_grasp_50g_1128_force_xyz.png)\n\n![第二轮曲线](../outputs/nov20/ax50_pebble07/vector_fixed_grasp_v2_50g_1128_curves.png)\n')
(r/'docs/TASK07_AX50_FIXED_CONTACT_GRASP_20261128.md').write_text(text)
main=r/'docs/TASK07_AX50_PEBBLE.md';head=f'''## 2026-11-28 修复模型后 50 g 夹持/2 mm 探提：未通过

完成两轮实际复验，摩擦和穿透退让保护保留。首轮手腕升 {a['final_wrist_rise_mm']:.3f} mm，石子最大变化 {a['max_lift_mm']:.3f} mm，未稳定离地；提拉后食指平均净力向下约 0.176 N，托盘仍承重。第二轮增加承载余量启动门槛，未获得通过结果。接触平均值不能代替承重能力；原 20 mm 验收未变。视频、三维力、软垫形变与穿透记录见 [本轮报告](TASK07_AX50_FIXED_CONTACT_GRASP_20261128.md)。

''';main.write_text(head+main.read_text());print('reports updated')
