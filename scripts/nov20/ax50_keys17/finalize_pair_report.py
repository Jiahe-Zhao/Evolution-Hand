from pathlib import Path
import json
root=Path('/home/zjh/Evolution_PC');out=root/'outputs/nov20/ax50_keys17';r=json.loads((out/'repaired_four_v2_result.json').read_text());summary=json.loads((out/'repair_summary.json').read_text());ok=r['passed'];status='固定初态通过' if ok else '未通过'
p=out/'PAIR_REPAIR_REPORT.md';s=p.read_text();title='## 最终连续序列复验'
if title in s:s=s.split(title)[0]
s=s.replace('单指测试运行中；四指依序、误触检查及接触归属尚未通过，不能沿用旧手结果。','该段为早期状态，最终连续序列结果见下文；不能沿用旧手结果。').replace('第二轮运行中。','第二轮单指已通过：峰值2.836 mm，释放0.187 mm，其他键峰值0.227 mm；记录来自食指远端软垫。')
s+='\n'+title+'\n\n状态：**'+status+'**。来源 repaired_four_v2，一次连续6.5秒Newton仿真。按压/释放条件仍为绝对行程≥2 mm / ≤0.5 mm，四指各自对应一键。\n\n'
s+='小指旧姿态的实际关节跟踪误差最高约0.595°，但碰撞表面高度约96.87 mm，只能压键约0.8 mm。旧IK代理点目标不足以保证实际表面下压。最终保留原关节动作，在无名指完全释放后的4.9–5.1秒先将腕部额外垂直接近2 mm，然后保持腕目标，由小指下压；额外接近阶段的10 ms记录中四键均未触发（分阶段子步监测另核对邻键峰值）。未使用未通过几何可达检查的 left_plan_little_v2 方案。\n\n'
s+='|指定手指|按压阶段子步峰值 mm|释放末值 mm|触发时指定手指接触记录数|其他手指接触标签|\n|---|---:|---:|---:|---|\n'
for name,c in zip(['食指','中指','无名指','小指'],r['checks']):s+=f"|{name}|{c['peak_press_mm']:.4f}|{c['release_mm']:.4f}|{c['assigned_finger_contact_rows']}|{c['other_contact_labels']}|\n"
s+='\n行程每个物理子步监测：接触阶段dt=0.0000625 s；未触发键/接近阶段均检查，键行程无非有限记录，保存的刚体位姿与接触力亦均为有限值。软体稳定性另见空中专门测试。接触归属记录采样10 ms，仅统计该键达到触发阈值时的记录；不将接触点数当作独立重复试验次数。CSV中力为作用在手上的世界坐标力，软接触由VBD读回，刚接触按Newton shape0/shape1方向换算。\n'
s+='\n证据：repaired_four_v2.mp4（2倍慢放）、repaired_four_v2_travel.png、repaired_four_v2_contacts.csv、repaired_four_v2_result.json、repaired_four_v2_substeps.json。全局键行程峰值（10ms采样）mm：'+str(r['peak_mm'])+'。\n'
s+='\n修复前后同时间窗0.6–0.7秒对比见 common_window_comparison.json；软体最大速度2.040→0.170 m/s，局部微振仍在。修复后腕部保持段软体最大位置偏差0.0957 mm；因此只报告本次动作稳定及任务验收，不宣称所有姿态或其他19场景已适配。官方资产几何未修改，小指覆盖层—软掌碰撞保留。\n'
s+='\n复现：在项目根目录运行 `PAIR_KD=0.5 KEY_KD=0.5 RUN_TAG=repaired_four_v2 /home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python scripts/nov20/ax50_keys17/repaired_four_v2.py`。重复运行会覆盖同名输出，应改RUN_TAG保留本次证据。\n'
p.write_text(s)
d=root/'docs/TASK17_AX50_KEYS.md';s=d.read_text();marker='## 2026-11-24 AX50接触修复后的连续四指验收'
if marker in s:s=s.split(marker)[0]
s+='\n\n'+marker+'\n\n**'+status+'**，以此次连续序列为最新状态。小指覆盖层—软掌接触保留，修正混合阻尼；软指垫—轻质键帽另配接触阻尼并缩小时间步。键帽尺寸、质量、弹簧、关节阻尼、间距及触发/释放门槛不变。先完成空中屈指和5 mm/5°腕部诊断，再完成单指、四指复验。\n\n详见 [完整修复与验收报告](../outputs/nov20/ax50_keys17/PAIR_REPAIR_REPORT.md)。\n'
d.write_text(s)
d=root/'docs/SCENES_20_20261116.md';s=d.read_text();old='已完成四键共享物理世界导入、重力静置和逐键机构回弹诊断。实际单指和四指验收进行中，尚未通过；阶跃力诊断存在0.240 mm限位超调，保留问题记录。原20项历史通过不转移到AX50。详见 [AX50按键记录](TASK17_AX50_KEYS.md)。'
new='AX50 17号单指及连续四指序列'+status+'（2026-11-24，固定初态；碰撞保留，修正软硬体接触阻尼和时间步，小指前增加2 mm无触发预接近）。四键原尺寸、质量、弹簧、行程和验收门槛不变。此前机构阶跃力0.240 mm限位超调记录仍保留；本次不增加场景编号，其他场景的旧手结果不转移到AX50。详见 [AX50按键记录](TASK17_AX50_KEYS.md)。'
if old in s:s=s.replace(old,new)
else:s+='\nAX50 17 最新状态：'+status+'，见 [AX50按键记录](TASK17_AX50_KEYS.md)。\n'
d.write_text(s)
print('Updated report and two docs:',status)
