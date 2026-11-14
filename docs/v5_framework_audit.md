# v5 终局框架复查

诊断范围：不调整控制器、物体初始位置、成功条件或工作空间阈值。

- 修复终局探针统计：原先在 env.step 返回后读取 streak；完成环境已经自动 reset，会少记终局样本。现在从 reset 前 info.physics.streak 取值。此前的成功布尔值和奖励没有被这个显示问题改写。
- 211/grasp：旧断言只报告皮肤观测集合中存在非有限值。诊断会报告异常字段、环境、身体、分量以及区域和根部原始位姿/速度，不用有限值替换异常。首次复现之前不判定物理或观测变换是根因。
- 111/strike 重放：第 49 步，4 个环境均未持握，手指对工具力均为零；工具相对目标 XY 距离为 0.20093 m（半径 0.18 m），Z 约 0.072 m，连续 6 步越界。正常触发 XY 越界终止，非超时。没有由此证明工具脱离的物理原因。
- 复测输出：outputs/v5_acceptance/framework01_{111,211}_strike/strike_terminal.json；framework01_111_branch/successful_terminals.json；framework01_211_grasp.log。仅有独立落盘证据才算结论。
- 211/strike 重放：第 53 步，XY 距离 0.20585–0.20888 m，Z 为 0.08785–0.09041 m；四个环境连续 6 步越界，终局未持握且手指对工具力为零。两手的终止与相同判据一致。

## 复查快照（重放尚在运行）

- framework01_211_grasp：当前落盘 306 步，尚无新终局或异常栈。
- framework01_111_branch：当前落盘 271 步，尚无新终局或异常栈。
旧 branch 第 450 步：env 0 成功、terminated=true、truncated=false、奖励 10.440855；旧连续计数为 reset 后读取，不能替代终局快照。旧 grasp 只保留了集合断言，无法从历史文件判定第一个非有限字段。
