# hand_111 毛笔书写同济（2026-11-22）

按用户要求暂停 AX50 适配，回到上一版 hand_111。15 笔物理动作完整执行，90.827 s；从预置执笔姿态开始，不包含拾笔。笔杆自由，控制只作用手腕和手指，沿用 13 束、65 段物理毛束和 900 根显示毛丝。实际受力接触才落墨。

状态：已完成书写演示，精度验收未通过。逐笔 6/15 通过（覆盖率至少 90%、RMS 不超过 3 mm，四指接触要求不变）。空中墨点 0；求解器警告 0；纸面峰值反力 0.8286 N；笔毫纸面最大压入 0.0363 mm；笔杆最大倾斜 13.57 度。后半段偏移与转折跟踪仍待改进，不将演示完成标为验收通过。

视频为 2 倍速，左侧完整执笔、右上笔毫近景、右下纸面俯视。仅对渲染墨点按 0.2 mm 网格保留首个接触点去重，原始物理记录及验收数据不变。

[三视角视频](../outputs/nov20/brush14/tongji_hand111_dense_brush_views.mp4)；[轨迹叠图](../outputs/nov20/brush14/tongji_hand111_overlay.png)；[验收明细](../outputs/nov20/brush14/tongji_hand111_writing_metrics.json)；[接触记录](../outputs/nov20/brush14/tongji_hand111_contacts.csv)。

复现：MODE=hao DT=.00005208333333333333 .venv/bin/python scripts/nov20/brush14/run_tongji.py tongji_hand111
