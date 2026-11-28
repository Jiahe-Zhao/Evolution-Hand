# 16 拨闩开盖：AX50固定初态通过
更新：2026-11-25。当前结果：完整 4.10 s 物理轨迹通过；未验证随机初态或实机。

- 原物体、关节与验收保留：固定中空盒身、被动铰链盖（`lid_hinge`, 0..1.66 rad）、外露拉环、盖耳及独立被动直线滑闩（`slide_bolt`, 0..18 mm，锁舌与盖耳初始侧向重叠 8 mm）。无开盖弹簧，正反例均仅驱动 AX50 手部关节，不驱动盖或闩。
- Newton MuJoCo-Warp + VBD 耦合仿真，AX50 v3.0.0 左手（`six_axis` 腕部，掌心朝上挂载），保留已通过 17 的精确附着过滤及 `palm-skin5_1` / `palm_tile_12,14` 局部阻尼修正；物体软刚接触采用 `ke=500, kd=0.05`；拨闩段 `dt=0.000125 s`，拉盖段 `dt=0.0000625 s`。

|同一模型的试次|实测|
|---|---|
|机构力矩对照，锁住 / 初始预退闩 18 mm（纯物体 GPU 探针）|同施 0.025 N·m 开盖力矩（前 30 步线性加载，共 240 步），最大开角 1.86° / 108.05°；后者末帧 95.61°，全程有限|
|锁住，跳过拨闩，直接用手拉环（`direct` 反例）|盖耳接触锁舌（峰值 0.268 N）；锁闩最大位移 0.00 mm（<1 mm），最大开角 1.83°（<3°），全程有限|
|手先拨闩再开盖（`hand` 正例）|食指拨动滑闩，1.44 s 退过 8 mm（解锁前最大开角 0.19° <3°），终态滑闩 15.69 mm；随后食指勾住拉环配合屈指与腕部缩回补偿上拉，3.13 s 首次达到 60°；最大开角 98.10°，末帧 98.03°，全程有限。**通过**|

- 证据：`outputs/nov20/ax50_latch16/` 下 `hand.mp4`、`direct.mp4`（左：AX50 完整场景；右：同帧隐藏手显示机构）、`hand_curves.png`、`direct_curves.png`、`verify_result.json`、`object_probe.json`、`hand_trace.csv`、`direct_trace.csv`、`hand_contacts.csv`、`direct_contacts.csv`、`hand.npz`、`direct.npz`。
- 脚本：`scripts/nov20/ax50_latch16/` 下 `latch_runtime.py`、`object_probe.py`、`run_latch16.py`、`render_latch16.py`；使用 `.venv/bin/python` 运行。
