# 抓握 BC / PPO 复用入口（2026-10-22）

当前验证通过的 BC：`bc_contraction29/bc_initial.pth`，固定机械手、80 g 自由工具、抓握/抬升/轻微扰动恢复。独立种子 63/64 成功。该结论不覆盖击打、四任务或形态变异。

- [实验结果与根因](outputs/wrap_learning_repair_20261022/ROOT_CAUSE_AND_RESULTS.md)
- [跨场景复用检查流程](outputs/wrap_learning_repair_20261022/REUSE_GUIDE.md)
- [复现命令与依赖](outputs/wrap_learning_repair_20261022/REPRODUCE.md)
- [模型选择与训练状态](outputs/wrap_learning_repair_20261022/selected_models.json)

远端实验目录：`/home/zjh/Evolution_PC/outputs/wrap_learning_repair_20261022`。

策略使用 67 维观测、24 维位置目标动作。迁入正式环境前必须保持动作单位、坐标、关节顺序、控制频率和观测编码一致，并在正式入口重新验证。不要把输出目录中的任意 `ppo_final.pth` 当作合格模型；最终选择以模型状态文件为准。

本次没有修改手结构。最终 PPO 接续未通过：同组成功率从 244/256 降到 74/256，独立种子为 22/64。因此未启动正式长训，旧多代进程保持暂停。BC29 的小幅随机动作测试为 63/64，保留为后续接续研究的起点。
