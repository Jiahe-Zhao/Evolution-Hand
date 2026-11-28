
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import json,sys
from pathlib import Path
root=_EVOLUTION_ROOT;out=root/'outputs/nov20/verified_objects'
def js(p):return json.loads(p.read_text())
p=js(out/'pebble/result.json');key=sys.argv[1];s=js(out/key/'result.json');a=js(out/'pebble/metrics.json');b=js(out/key/'metrics.json')
region=sorted(set(b[-1]['tiles']));rst='通过' if s['success'] else '未通过'
body=f"""## 关节动作正例复验

hand_111 左手；原场景和原 Score，Newton/MJWarp CUDA，步长 1/240 s，720 步。两项均从标准初态开始，控制端只写手部关节目标。固定初态的轨迹见证，不是策略成功率。

|任务|实测|判定|
|---|---|---|
|07 捏取石子|最高抬升 {p['max_lift_mm']:.2f} mm；邻石最大位移 {p['neighbor_mm']:.2f} mm；双指正文条件连续成立 {p['maximum_streak']}/240 = {p['maximum_streak']/240:.3f} s|通过，首次成功 {p['first_success']:.3f} s；末端没有持续持稳到回合结束|
|08 刮削附着层|首块完成前，手接触握部且工具不碰托架时，最高抬升 {s['held_off_rest_before_first_patch_mm']:.2f} mm；完成 {s['cells']}/24 区，要求 20 区|{rst}|

07 为单环境回放。08 为 {s['batch']} 个并行环境中的第 {s['lane']} 号（从 0 起），同一环境、同一条连续轨迹。视频从该次 Newton 仿真的关节状态渲染，显示两机位；没有拼接不同轨迹。使用原 Score 对该环境的工具接触约束、逐接触解算力及工具位姿复算。手接触握部不等于全程无滑脱，未测量去除质量。

- 石子：[操作视频](../outputs/nov20/verified_objects/pebble/operation.mp4)、[逐步指标](../outputs/nov20/verified_objects/pebble/metrics.json)。
- 刮削：[操作视频](../outputs/nov20/verified_objects/{key}/operation.mp4)、[逐步指标](../outputs/nov20/verified_objects/{key}/metrics.json)。
- 刮削原始证据：{s['source']}；同目录 best 文件保存动作。原始计分模块 SHA-256：{s['source_score_sha']}。
"""
md=root/'docs/OBJECT_TASKS_07_08.md';txt=md.read_text().split('\n## 关节动作正例复验')[0].replace('两项完整正例待验收。','关节控制实测见文末。')
md.write_text(txt.rstrip()+'\n\n'+body.replace('双指正文条件','双指捏取、排除旁撑及抬升条件'))
f=root/'docs/SCENES_20_20261116.md';txt=f.read_text().split('\n本批正例状态（关节动作）')[0];f.write_text(txt.rstrip()+f'\n\n本批正例状态（关节动作）：07 通过；08 {s["cells"]}/24，{rst}。详见 [实测与视频](OBJECT_TASKS_07_08.md)。\n')
(out/'RESULTS.md').write_text(md.read_text())
