# 自由工具 Strike 新场景与训练（2026-10-20）

## 场景及脚本
用户明确选择保持自由工具，不使用轴向导向或固定约束。新场景预设为 `open_grasp_v4`：工具仍是自由刚体，截面由 25×25 mm 改为 18×18 mm，长度仍为 180 mm；归一化初始屈曲由 0.45 改为 0.0（更张开）。保留原质量、物理成功阈值、20 秒回合、抓握恢复及初始碰撞间隙检查。没有修改机械手 URDF、关节轴、结构网格或限位。

新脚本通过正式策略动作逐指缓慢闭合：接触不足时继续闭合、接触建立后保持、过大接触力时微调。稳定抓握后等待 40 步；工具已在目标区域内时只做竖直下压，避免多余横向移动破坏抓握。脚本不在动作阶段写入工具位姿或固定工具。

Strike 固定种子 7、8、9 均通过完整抓握和击打验收。seed 7 用 1125 步完成，最大关节越限为 0。该结果是脚本验收，不代表 BC/PPO 策略已达到相同成功率。

## 训练与测试一致性
- `mechanical_scene_contract.py` 共享物理配置；Strike 契约版本为 4。
- 正式训练、脚本、BC 和评估均设置：
  - `EVOLUTION_STRIKE_SCENE_PRESET=open_grasp_v4`
  - `EVOLUTION_STRIKE_V4_TOOL_WIDTH_M=0.018`
  - `EVOLUTION_STRIKE_V4_PREGRASP=0.0`
- 独立测试入口：`bash Isaaclab_other/evaluate_free_tool_v4.sh <原评估参数>`。
- 保留严格场景契约检查，不用修改旧检查点元数据来伪装场景一致。
- 其余三任务重新复验并重新生成匹配当前实现哈希的 BC，避免混用旧场景轨迹。

## 保留的正式参数
4096 环境；PPO horizon=16、minibatch=4096、mini_epochs=5、learning_rate=5e-4。15 代、种群 8、初始种群 8；两阶段预算 100/250，Forage 25/50，Strike 100/250。BC 保持 200 个监督 epoch。新实验不覆盖旧 lineage、日志或 checkpoint，旧暂停进程未恢复。

## 新实验与文件
- 实验：`exp_20261020_free_tool_v4_BC15g_v2`
- 启动入口：`Isaaclab_other/run_formal_free_tool_v4_20261020.sh`
- 完整准备流程：`Isaaclab_other/start_free_tool_v4_pipeline.py`
- 新脚本：`evolution_tasks/task_suite/mechanical_strike_v4_demo.py`
- 数据、BC、启动状态、视频：`outputs/strike_v3_training_20261020/`（目录沿用准备阶段名称，实际 Strike 场景版本为 v4）。
- 运行状态：`pipeline_status.json`；只有其中记录 `training_pid` 才表示已提交正式训练启动。

本文件将追加正式启动及运行证据；前置脚本、BC 或 4096 环境检查失败时，准备流程不会启动训练。

## 本次实际执行记录
- Forage 脚本原先引用已删除的两个叶片传感器，现按统一传感器的 `[环境, 手部刚体, 叶片, xyz]` 接触矩阵读取选定指节与叶片接触；物理任务未改。修复后 seed 7 完整通过，无关节越限。
- Grasp、Branch、Forage 各 1 个种子脚本通过，Strike 3 个种子通过。
- 四任务 BC 已生成；最终监督损失：Grasp 6.603e-5、Branch 1.173e-4、Forage 4.646e-5、Strike 1.626e-6。损失低不等于自主任务成功。
- Strike BC 自主测试 seed 7 成功率 0%，保留这一结果，继续 PPO 学习。
- 4096 环境运行 300 步通过：状态有限，局部重置对其他环境位移影响为 0。
- 正式进程 PID 3719583，日志 `/tmp/formal_free_tool_v4_20261020.log`，已开始初始种群形态物理检查。PPO 迭代进度另行追加。
- 保留旧训练暂停状态，没有终止、覆盖旧实验。
