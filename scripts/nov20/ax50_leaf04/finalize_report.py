
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,csv
root=Path((str(_EVOLUTION_ROOT)));out=root/'outputs/nov20/ax50_leaf04';r=json.loads((out/'hand_v7_result.json').read_text());assert r['passed'];rows=list(csv.DictReader((out/'hand_v7_trace.csv').open()));a,b=r['hold_interval_s'];hold=[x for x in rows if a-1e-7<=float(x['time_s'])<=b+1e-7];assert all(int(x['qualified_substeps'])==int(x['substeps']) for x in hold)
visible=list(csv.DictReader((out/'hand_v7_visibility.csv').open()));vv=[float(x['visible_fraction']) for x in visible if a-1e-7<=float(x['time_s'])<=b+1e-7]
summary=f'''# 04 拨叶翻找：AX50固定初态通过
更新：2026-11-25。当前结果：hand_v7，完整4.9秒物理轨迹通过；未验证随机初态或实机。

- 原物体、目标、相机和验收保留；两叶独立被动铰链，手部正例无叶片驱动或额外探针力。
- 食指先拨并保持第一片，中指再拨第二片。{a:.3f}–{b:.3f}秒连续{r['qualified_hold_s']:.3f}秒满足全部条件，可见率范围{min(vv)*100:.2f}%–{max(vv)*100:.2f}%。固定视角2240条射线包含AX50显示皮肤、实际软体表面、叶片和叶柄。
- 该段逐物理子步两指定末节最小法向载荷为{r['qualified_interval_min_tip_N'][0]:.6f}/{r['qualified_interval_min_tip_N'][1]:.6f} N，非指定手部对叶最大载荷{r['qualified_interval_max_unassigned_N']:.6g} N。每10ms窗口保存全部子步最小/最大载荷及合格步数；原始逐接触点记录为100Hz，视频半速显示。
- 逐子步GPU汇总与独立Python端点汇总最大差{r['audit_endpoint_max_error_N']:.3g} N。全程{sum(int(x['substeps']) for x in rows)}个审计子步、状态有限；dt在预屈指前0.5秒为0.000125秒，其后为0.0000625秒。
- 撤手后两叶终态{r['final_leaf_degrees'][0]:.3f}/{r['final_leaf_degrees'][1]:.3f}°，均小于1°，目标重新遮挡；末段手–叶载荷归零。

修复内容：规划点改到正确接触侧，按实际末节边界选点，并对中节加入避让；仅针对软指垫–叶片接触使用ke=500、kd=0.05的数值罚参数，保留碰撞、摩擦、弹性反力和反馈。叶片质量/几何/被动关节未改，官方AX50资产及已通过17入口未改。该参数针对本场景轻叶片，不作为所有物体的统一值。

证据：outputs/nov20/ax50_leaf04/hand_v7.mp4、hand_v7_curves.png、hand_v7_result.json、hand_v7_trace.csv、hand_v7_contacts.csv、hand_v7_visibility.csv、hand_v7.npz；对照contact_probe_comparison.png及三个hold探针JSON/CSV。代码及规划快照reproduction_v7/。

## 调试历史（以下失败状态均为历史记录，以顶部当前结果为准）

'''
p=root/'docs/TASK04_AX50_LEAF.md';old=p.read_text();p.write_text(summary+old)
p=root/'docs/AX50_MIGRATION_PLAN.md';s=p.read_text().replace('2. 04 拨叶翻找：当前进行，两个独立被动铰链、双指保持、固定视角遮挡。','2. 04 拨叶翻找：固定初态通过（2026-11-25），完整结果见 TASK04_AX50_LEAF.md。').replace('3. 13 活动提梁：勾持铰链与提起器身。','3. 13 活动提梁：下一项；对象导入核对完成，手部动作未验证。');s=s.replace('## 当前04','## 04适配早期记录（历史）');s='当前AX50复验通过：2/20（17、04）。13仅完成资产核对，07及后续尚未通过。\n\n'+s;s+='\n2026-11-25最新：04 hand_v7通过。新增经验：薄且轻的物体需按接触数量与等效质量检查软刚耦合罚刚度；不能把17或04的数值参数无条件复制到其他物体。逐子步载荷审计已接入，不用帧级平均力掩盖断触。\n';p.write_text(s)
p=root/'docs/SCENES_20_20261116.md';s=p.read_text();s+=f'''\n## AX50 04 拨叶翻找：固定初态通过（2026-11-25）
hand_v7原物体及验收不变，食指保持第一片、中指拨第二片；连续{r['qualified_hold_s']:.3f}秒固定视角可见率≥80%，两指定末节逐物理子步载荷均>0.01 N，无其他手部有效承力。末态两叶均约-0.176°，撤手回落正常。详见 [AX50 04结果](TASK04_AX50_LEAF.md)。本次使用目标接触点/中节避让规划及仅针对软指垫-叶片的ke=500、kd=0.05数值罚参数，保留碰撞和反力。
AX50已复验通过2/20（17、04）；原hand_111历史20场景记录不变。13活动提梁完成对象导入核对，尚未做AX50手操作；随后07捏取石子。03搬运、18手机点选继续暂缓。
''';p.write_text(s)
(out/'HAND_V7_REPORT.md').write_text(summary.split('## 调试历史')[0]);print(summary.split('## 调试历史')[0])
