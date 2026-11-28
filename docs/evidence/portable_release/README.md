# 便携发布实测（2026-11-28）

## 验证方式

运行代码提交见 environment.json 的 source_commit。两个空目录分别从该 Git 提交导出，仅使用已跟踪文件进行 bootstrap；没有复制实验目录的 outputs。依赖安装在独立 Python 3.12 venv，先使用固定版本 wheels 安装，再执行 bootstrap --install 和 pip check，均通过。系统驱动和 Warp 编译缓存复用；不是另一台全新操作系统的测试。

## 结果

| 检查 | 结果 | 范围 |
|---|---|---|
| 基础刚体入口 | 22/22 | 模型/网格加载、24 步有限值检查 |
| AX50 左右手入口 | 16/16 | 每组 400 步（0.05 s），状态有限、reset 误差 0、reset 后继续推进 |
| 基础完整动作示例 | 22/22 正常退出 | 按场景命令跑完；不代表 22 个目标均成功 |
| 旧展开机构左右手 | 2/2 | URDF/网格加载和 FK 有限；未认证受载接触 |
| 工具单元检查 | 2/2 | 负奖励文件名解析；左右手网格引用 |
| Python 语法检查 | 通过 | 447 个场景脚本及统一入口 |
| 39 项保留补丁 | git apply --check 通过 | 未应用，不声称旧 PhysX 回归通过 |

AX50 16 组使用无控制器的短程装配姿态，不能替代长时间接触、任务成功率或 PPO 训练验证。刮擦 08 在本次 CPU 单场景完整回放中 success=false、cells=0；与原 CUDA 并行历史成绩不同，未修改动作来掩盖这一点。各场景实际输出见 clean_demo/*.log。

## 文件访问核查

两个完整 suite 同时启用 Python open 审计和 Linux strace 文件系统调用跟踪。没有打开旧实验项目、原 IsaacLab 环境或实现工作区的文件。仅出现审计器解析禁止目录本身时产生的 stat/lstat；没有访问这些目录下的文件。原始 trace 的 SHA-256、字节数和计数见 file_access_audit.json。原始 trace 留在验收机器，不作为运行依赖。

clean_all/ 和 clean_demo/ 保留汇总与逐项记录；日志中的本次临时 checkout 前缀替换为 <validation_checkout>，方便查看。环境版本、GPU/驱动见 environment.json。

## 修复与边界

补齐此前被忽略的命名种子，修复 XML 子树搬移后的网格引用；敲击装配恢复模块原定手掌朝向。AX50 v3 ZIP 未变，未改手型、材料、碰撞和控制器。原实验 checkout 的 47 处未提交修改仍保留；发布侧分别按 6 项合入、2 个独立模型、39 项保留补丁处理。
