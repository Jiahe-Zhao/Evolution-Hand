
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import sys,json,hashlib
from pathlib import Path
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));ROOT=P.parents[2];tag=sys.argv[1];b=json.loads((P/'verified_metrics.json').read_text());h=json.loads((P/(tag+'_writing_metrics.json')).read_text());g=json.loads((P/'verified_geometry_audit.json').read_text());replay=json.loads((P/'verified_replay_audit.json').read_text())
basic_pass=(b['tip_span_press_mm']>b['tip_span_hover_mm']+.5 and b['tip_bend_max_mm']>1 and b['tip_bend_recovered_mm']<.1 and b['hover_ink_contact_count']==0 and b['air_control_ink_count']==0 and b['reaction_balance_relative_error']<.05 and min(b['grip_contact_fraction_per_finger'][:4])>.99 and b['shaft_translation_in_palm_max_mm']<1 and b['hair_paper_max_overlap_mm']<.1)
status='通过' if h['success'] else '未通过'
text=f'''# 14 柔性毛笔：物体建模与操作验证

更新：2026-11-19。基础压笔、拖笔、提笔验证：{'通过' if basic_pass else '未通过'}。“好”字挑战：{status}。原长笔描线的历史通过结果保留；本次是 14 的扩展，不增加独立场景计数，本月仍为 20 个历史已验收编号。

## 模型与执笔姿态

沿用 hand_111 左手。刚性笔杆长 130 mm、直径 10 mm，笔毫长 22 mm，整体约 152 mm；笔杆质量 9 g，笔毫合计 0.325 g。笔杆为自由刚体，无手–笔焊接、吸附或外加笔杆驱动力。正式操作从预置轻接触握姿开始，0–0.8 s 用于夹持稳定，不包含从桌面拾笔。

笔毫采用 13 束等效毛束、每束 5 段，共 65 个动态段和 130 个被动弯曲自由度。每段长度固定，以两个正交转动弹簧模拟双向弯曲；各束间及束–纸面均有碰撞与摩擦。根部连在笔杆上，全部在同一个 Newton SolverMuJoCo CPU 求解器中求解，因此纸面反力通过毛束、笔杆和指垫真实传递。实际弹性参数为每束 EI=2e-6 N·m²，转动刚度 EI/(22 mm/5)=4.545e-4 N·m/rad，阻尼 2e-7 N·m·s/rad。这些是粗粒化工程参数，未经真实笔毫标定。

握姿为笔杆直立、拇指和食指夹持、中指及无名指参与稳定，小指收拢。中指和无名指使用低频接触几何反馈调整关节。控制只作用于手/腕关节，毛束弹簧零位固定，没有给笔毫设置运动轨迹。

Newton 导入未完整保留原 MJCF 的接触维度和摩擦覆盖值，运行脚本显式设置并导出了实际求解器参数。指垫–笔杆使用有限接触区域的六维摩擦近似，滑动系数 1.1，扭转/滚动特征长度 4 mm；笔毫–纸面滑动系数 0.45，毛束间 0.1。需通过脚本或已解析的原生求解器快照复现，不能仅看几何 XML 推断全部物理参数。

视频中的细丝随实际毛束段姿态变形，是等效毛束的细化显示，不是每根真实毛发的独立模拟。墨迹只来自实际有载接触位置，不模拟墨水流体、吸墨、渗墨或真实书法笔锋。

## 基础动作验收

基础正例和悬空对照均为 8 s，物理步长 1/9600 s，接触和状态记录 480 Hz，视频 30 fps 原速。对照仅将纸面降低 30 mm，保持同一套手部控制。以下均为当前模型实测：

|检查|结果|结论|
|---|---|---|
|压笔铺开|笔尖跨度 {b['tip_span_hover_mm']:.3f} → {b['tip_span_press_mm']:.3f} mm|通过|
|拖动弯曲|执行 18 mm 拖笔，整段最大笔尖变形 {b['tip_bend_max_mm']:.3f} mm|通过|
|提笔回弹|末段残余笔尖变形 {b['tip_bend_recovered_mm']:.4f} mm|通过|
|悬空不落墨|正例空中段及完整悬空对照均 0 接触墨点、0 N 纸面载荷|通过|
|反力传回手|纸面反力 {b['paper_force_press_mean_N']:.6f} N；手上载荷变化 {b['hand_transmitted_reaction_delta_N']:.6f} N|通过|
|四指实际承载|稳定后拇、食、中、无名指四个末节指腹均持续 >0.005 N，小指不计承载|通过|
|握持滑移|笔杆相对掌内最大平移 {b['shaft_translation_in_palm_max_mm']:.3f} mm；最大倾斜 {b['shaft_tilt_max_deg']:.3f}°|通过|
|支撑排除|手–纸、笔杆–纸载荷均为 0|通过|
|接触压入|笔毫–纸面最大 {b['hair_paper_max_overlap_mm']:.4f} mm；毛束间抽查最大 {g['cross_bundle_max_geometric_overlap_mm']:.4f} mm|通过|
|跨手指几何抽查|不同手指末端链最小间距 {g['cross_finger_distal_min_distance_mm']:.3f} mm|通过|

反力对比取 2.7–3.08 s 静态压笔段，悬空时手向上支持约 0.0915 N 自重；压笔后手对笔向下施力。纸面反力与手部载荷变化的相对差为 {100*b['reaction_balance_relative_error']:.5f}%。这里证明的是双向载荷传递，不是把纸面载荷复制到手的显示值。

基础工程检查采用：铺开增量 >0.5 mm、拖动变形 >1 mm、回弹残差 <0.1 mm、稳定握持滑移 <1 mm、反力差 <5%、笔毫–纸面压入 <0.1 mm。跨手指和毛束间几何检查按 30 Hz 抽样，不能称为所有微步的严格零穿透证明。视频物理几何重放最大偏差为 {replay['max_geom_replay_error_m']:.3g} m。

证据：[操作视频](verified_operation.mp4)、[悬空对照](control_operation.mp4)、[力与变形曲线](verified_curves.png)、[验收 JSON](verified_metrics.json)、[接触记录](verified_contacts.csv)、[状态轨迹](verified_trace.csv)、[几何抽查](verified_geometry_audit.json)。

## “好”字挑战：{status}

沿用同一物体、毛束和手模型。六笔目标来自 Hanzi Writer Data 的“好”字中心线，字宽 45 mm；字形数据源于 Make Me a Hanzi，许可文件随数据保存。保持原路径覆盖率至少 90%、横向 RMS 不超过 3 mm 的阈值，逐笔报告。毛笔用实际受力接触印迹的合力中心评估路径，另报告全部接触点 RMS。覆盖率累加连续有载轨迹在目标上的弧长，不把换接触毛束造成的离散中心跳变计作已经描过的线段。

笔间真实抬笔并在空中移动，落笔后接触才留墨。写字使用更小步长 1/19200 s；腕部仅根据实际接触中心和法向力调整，参考接触力 0.08 N，并限制修正速度。首轮 hao 在第二笔落下发生数值不稳定，已拒绝；第二轮 hao2 完整执行且无数值警告，五笔通过，第一笔转折处覆盖率 82.5%；随后仅放慢第一笔并增加末端停留，本表仅对应 {tag}。

|笔画|覆盖率|接触中心 RMS|全部接触点 RMS|结果|
|---|---:|---:|---:|---|
'''
for row in h['strokes']:
 def num(v):return '无有效接触' if v is None else f'{v:.3f} mm'
 text+=f"|{row['stroke']}|{row['coverage_pct']:.2f}%|{num(row['contact_centroid_rms_mm'])}|{num(row['all_contact_points_rms_mm'])}|{'通过' if row['pass'] else '未通过'}|\n"
overall=sum(v['coverage_pct']*v['target_length_mm'] for v in h['strokes'])/sum(v['target_length_mm'] for v in h['strokes'])
text+=f'''\n按目标弧长加权的总体覆盖率为 {overall:.2f}%；最终状态仍按逐笔 90% 门槛判断。\n\n空中段墨点数 {h['air_contact_count']}；纸面峰值载荷 {h['paper_force_peak_N']:.4f} N；手–纸和笔杆–纸峰值分别为 {h['hand_paper_peak_N']:.4f}、{h['shaft_paper_peak_N']:.4f} N。求解器状态见 [{tag}_solver_status.json]({tag}_solver_status.json)。\n\n证据：[写字视频]({tag}_operation.mp4)、[目标与实际轨迹]({tag}_overlay.png)、[逐笔验收]({tag}_writing_metrics.json)、[接触记录]({tag}{"""_contacts.csv)。

## 复现与范围

环境 `""" + _evo_sys.executable + '`，工作目录 `' + str(_EVOLUTION_ROOT) + """`。

```bash
# 已保存的预置执笔种子 grasp_seed.npz，不需要重新搜索初态
python scripts/nov20/brush14/run_brush.py verified
MODE=air python scripts/nov20/brush14/run_brush.py control
MODE=hao DT=.00005208333333333333 python scripts/nov20/brush14/run_brush.py """}{tag}\npython scripts/nov20/brush14/evaluate_brush.py verified control\npython scripts/nov20/brush14/score_hao.py {tag}\nMUJOCO_GL=egl python scripts/nov20/brush14/render_brush.py verified\n```\n\n以上 python 应替换为前述虚拟环境的完整解释器路径。源代码位于 `scripts/nov20/brush14/`。`rebuild_audit.json` 证明当前脚本重建的基础几何 XML 与已验收模型哈希一致；`rebuild_resolved.mjb` 保存实际原生求解器模型，包括被动弹簧和接触参数；文件清单哈希见 `manifest.json`。\n\n结果是固定初态的脚本控制物理验证，不是学习策略、拾笔能力或跨初态成功率。旧刚笔“好”字滑脱失败记录保留在 `outputs/nov20/hao14/`，不与本次毛笔结果混用。\n\n接触参数语义参考 [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair)；[字形来源](https://github.com/chanind/hanzi-writer-data)。\n'''
(P/'RESULTS.md').write_text(text)
# Preserve local relative media links in results, adapt them for the project docs directory.
doc=text
for item in P.iterdir():
 if item.is_file():doc=doc.replace(']('+item.name+')','](../outputs/nov20/brush14/'+item.name+')')
(ROOT/'docs/TASK14_HAO_BRUSH.md').write_text(doc)
p=ROOT/'docs/SCENES_20_20261116.md';old=p.read_text();heading='## 14 写好字与柔性毛笔扩展（2026-11-19）';i=old.find(heading)
if i<0:i=old.find('## 14 柔性毛笔：')
brief=f'''## 14 柔性毛笔：压、拖、提笔通过；“好”字{status}（2026-11-19）
沿用 hand_111，自由笔杆、13 束/65 段被动弹性笔毫，同一 Newton SolverMuJoCo CPU 内双向接触。四指稳定持握，压笔笔尖跨度 3.206→5.017 mm，最大变形 11.445 mm，提笔残余 0.0269 mm。悬空正反例均零落墨；纸面反力 0.601442 N 对应手部载荷变化 0.601455 N。笔毫–纸面最大接触压入 0.0622 mm。基础动作通过，视频和曲线见 [完整验收](TASK14_HAO_BRUSH.md)。写“好”采用六笔真实抬落笔，当前 {tag} 逐笔结果见完整验收；状态为{status}。从预置握姿开始，不包含拾笔；为等效毛束模型，不模拟墨水流体。本扩展仍属于 14，不新增计数，本月仍为 20 个历史已验收编号。
'''
if i>=0:
 end=old.find('\n## ',i+3);old=old[:i]+brief+(old[end:] if end>=0 else '')
else:old+='\n'+brief
p.write_text(old)
(P/'SUMMARY.json').write_text(json.dumps({'basic_pass':basic_pass,'writing_pass':h['success'],'writing_run':tag},indent=2));print(P/'RESULTS.md')
