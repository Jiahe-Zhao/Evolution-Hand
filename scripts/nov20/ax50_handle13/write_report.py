"""Publish measured task-13 results after successful complete-from-initial run."""
from pathlib import Path
import json,sys,re,hashlib,csv
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_handle13';tag=sys.argv[1]
r=json.loads((OUT/(tag+'_result.json')).read_text());m=json.loads((OUT/(tag+'_model.json')).read_text());c=json.loads((OUT/(tag+'_contact_summary.json')).read_text());render=json.loads((OUT/(tag+'_render.json')).read_text())
assert r['passed'] and r['stationary_hold']['ok'] and 'checkpoint_continuation' not in m
assert c['endpoint_normal_sum_max_error_N']<1e-4
h=r['hold'];a=h['all_substeps'];s=r['stationary_hold'];rows=list(csv.DictReader((OUT/(tag+'_trace.csv')).open()));steady=[x for x in rows if s['interval_s'][0]-1e-7<=float(x['time_s'])<=s['interval_s'][1]+1e-7]
lo=min(float(x['min_lift_m']) for x in steady)*1000;tilt=max(float(x['max_tilt_deg']) for x in steady);hinge=min(float(x['min_hinge_deg']) for x in steady)
text=f'''# 13 活动提梁：AX50 固定初态验收

更新：2026-11-25。**从原始初态完整复跑通过**，运行 `{tag}`。本次不从检查点恢复；4.5秒实际Newton动力学，手和物体在同一世界内运行。

## 原物体和验收

保留原 `outputs/nov20/handle.xml` 中器身、提梁、地面，item子树和floor逐项完全一致。器身80 g、提梁11 g；自由器身6自由度，提梁独立被动铰链1自由度，轴(0,-1,0)、范围0..2.7 rad、阻尼0.0008。横杆截面6×6 mm、长60 mm，距铰链64 mm。初始器身高度26 mm，提升以重力步进前的位置计算。

验收不变：提梁转起≥45°，仅通过提梁提起器身≥25 mm，倾斜≤15°，保持≥0.5秒；直接抓器身不算成功。

## 实测结果

|项目|结果|
|---|---|
|连续合格区间|{h['interval_s'][0]:.2f}–{h['interval_s'][1]:.2f} s，共 **{h['duration_s']:.2f} s**|
|该区间逐物理子步最小提梁转角|{a['min_hinge_deg']:.2f}°|
|该区间逐物理子步最小器身提升|{a['min_lift_m']*1000:.2f} mm|
|该区间逐物理子步最大倾斜|{a['max_tilt_deg']:.2f}°|
|手腕目标停止后的连续合格保持|{s['interval_s'][0]:.2f}–{s['interval_s'][1]:.2f} s，共 **{s['duration_s']:.2f} s**|
|静止保持段最低提升 / 最大倾斜 / 最小提梁转角|{lo:.2f} mm / {tilt:.2f}° / {hinge:.2f}°|
|全程手–器身法向力|{r['contact']['hand_item_max_N_anywhere']:.6f} N|
|合格区间逐子步最低手–提梁法向力|{a['min_bail_N']:.6f} N|
|全部记录状态有限|{r['checks']['all_finite']}|

姿态从记录body_q独立复算，Newton四元数按XYZW解读。验收检查每个物理子步，dt=0.000125 s（8000 Hz）；每10 ms汇总最小高度、最小转角、最大倾斜、载荷及合格子步数量。逐接触明细100 Hz，明细加总与GPU端点力统计最大误差 {c['endpoint_normal_sum_max_error_N']:.3g} N。手–器身有载排除阈值0.0005 N，提梁持续有载阈值0.01 N。

## 手部动作与接触

空中先后屈指，接近后转起提梁；先横向退开20 mm，随后下探45 mm，再横向移回42 mm，从横杆下方上提80 mm。只给AX50手关节目标，不驱动物体、不直接施加物体提升力、不焊接提梁。物体7个自由度位置驱动增益均0。

最终静止保持由中指末节 `link_3_3` 承载。无名指末节 `link_4_3` 在此前动作中有接触，静止保持段无有效承载。静止保持段平均手部反力向量为 {c['hold_mean_force_on_hand_N']} N。完整参与部位和占比见 `{tag}_contact_summary.json`，逐接触力见CSV。

复用17/04的27维动作接口、腕部/手指PD、已定位的掌部局部接触阻尼修正及附着重复约束过滤；保留有意义的自接触。没有改动物体尺寸、质量、铰链或关节限位。原XML包含旧手的816条专用接触对随旧手移除；新手和物体实际刚体mu均1.0，旧手显式手–物体接触对mu为0.9，因此只称对象保持，不称两代手的接触材料完全相同。未沿用04轻叶片的软接触ke=500。

## 失败与修复依据

- hook_preflight：空中屈指提前碰提梁；外移屈指位置45 mm后消除。
- handle_v1：转起最大88.55°，最大器身提升仅0.205 mm。横杆处于屈指外侧，侧推后脱开，未通过。
- handle_v2：直接下探时中指/无名指末节撞横杆，逐子步峰值约6.94 N，将提梁推过竖直并翻向另一侧；未通过。
- handle_v3_probe：先退开再下探，消除上述碰撞；从3.1秒全世界状态续跑验证动作，连续合格0.93秒。该记录仅作为调试依据。
- {tag}：从原始初态重新执行全部动作，并独立审计，作为本次正式结果。

## 交付与复现

- [三视角实际操作视频](../outputs/nov20/ax50_handle13/{tag}.mp4)：整手、器身离地近景、提梁接触近景；记录帧回放，2倍慢放，约{render['frames']/render['fps']:.1f}秒。数值显示底部净距，不添加类似支撑杆的三维测量线。
- [验收及接触力曲线](../outputs/nov20/ax50_handle13/{tag}_curves.png)
- [分部位接触力曲线](../outputs/nov20/ax50_handle13/{tag}_contact_curves.png)
- [逐窗口验收记录](../outputs/nov20/ax50_handle13/{tag}_trace.csv)
- [逐接触明细](../outputs/nov20/ax50_handle13/{tag}_contacts.csv)
- [机器可读结果](../outputs/nov20/ax50_handle13/{tag}_result.json)

Python：`/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python`。项目根目录执行：

```bash
RUN_TAG=handle_v3_repeat DURATION=4.5 SIM_DT=.000125 /home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python scripts/nov20/ax50_handle13/run_handle_v3.py
```

评估脚本：`audit_handle.py`、`contact_summary.py`、`render_handle.py`。这是固定初态物理轨迹验收，不代表随机初态成功率或实机验证。
'''
doc=ROOT/'docs/TASK13_AX50_HANDLE.md';backup=OUT/(tag+'_previous_task13.md')
if not backup.exists():backup.write_text(doc.read_text())
doc.write_text(text);(OUT/(tag+'_REPORT.md')).write_text(text)
p=ROOT/'docs/AX50_MIGRATION_PLAN.md';v=p.read_text();match=re.search(r'当前AX50复验通过：(\d+)/20（([^）]+)）',v);ids=match[2].split('、');ids=ids if '13' in ids else ids+['13'];count=len(ids);label=f"{count}/20（{'、'.join(ids)}）"
v=re.sub(r'^当前AX50复验通过：.*$',f'当前AX50复验通过：{label}。13活动提梁从原初态完整复跑通过；07及后续继续按计划推进。',v,count=1,flags=re.M)
v=re.sub(r'^3\. 13 活动提梁：.*$',f'3. 13 活动提梁：固定初态通过（2026-11-25），原物体及验收保留，静止保持{s["duration_s"]:.2f}秒。见 TASK13_AX50_HANDLE.md。',v,flags=re.M);p.write_text(v)
p=ROOT/'docs/SCENES_20_20261116.md';v=p.read_text();v=re.sub(r'AX50已复验通过\d+/20（[^）]+）；[^\n]*',f'AX50已复验通过{label}；原hand_111历史20场景记录不变。13活动提梁已从原始初态完整复跑通过，见 [13结果](TASK13_AX50_HANDLE.md)。03搬运、18手机点选继续暂缓。',v)
heading='## AX50 13 活动提梁：固定初态通过（2026-11-25）';v=v.split('\n'+heading)[0].rstrip()+f'\n\n{heading}\n\n{tag}完整复跑：连续合格{h["duration_s"]:.2f}秒，手腕目标静止后保持{s["duration_s"]:.2f}秒。静止段最低提升{lo:.2f} mm、最大倾斜{tilt:.2f}°、最小提梁角{hinge:.2f}°；全程手–器身接触力0，仅经提梁承载。模型与原验收不变。见 [视频、接触记录和验收](TASK13_AX50_HANDLE.md)。AX50当前{label}，不增加历史场景编号。\n';p.write_text(v)
files=list((ROOT/'scripts/nov20/ax50_handle13').glob('*.py'))+[OUT/'object.xml',OUT/'plan.json',ROOT/'scripts/nov20/ax50_leaf04/leaf_runtime.py',ROOT/'scripts/nov20/ax50_keys17/damped_adapter.py',ROOT/'scripts/nov20/ax50_keys17/pair_damping.py'];manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(OUT/(tag+'_manifest.json')).write_text(json.dumps(manifest,indent=2));print(label);print(s);print('stationary min lift, max tilt, min hinge',lo,tilt,hinge)
