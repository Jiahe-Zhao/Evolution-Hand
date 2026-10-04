# Evolution_PC 双 Worker 并行改造与验证记录

日期：2026-10-01

## 1. 本次处理范围

- 已停止旧实验：`exp_20260930_fast_stage1_150_worker4_v1`。
- 未删除旧实验结果、代码或资产；旧日志保留在 `evolution_tasks/logs` 中。
- 已确认停止后没有活动的 `main_evolution.py`、`train_worker.py` 或碰撞检查进程。

## 2. 已完成的代码修改

### 双 Worker 默认配置

文件：`Isaaclab_other/run_local_evolution_4090_15g_single_stage_reuse.sh`

- `EVOLUTION_PARALLEL_SLOTS` 默认改为 `2`。
- `EVOLUTION_PARALLEL_SPLIT_ENVS` 默认改为 `1`。
- 基础环境数为 4096 时，每个 slot 使用 2048 个环境。
- `EVOLUTION_ISAAC_WORKER_MAX_REQUESTS` 默认改为 `4`，每个 worker 连续处理 4 个请求后回收，降低长时间 native 状态累积风险。
- `EVOLUTION_TASKS` 改为支持外部环境变量覆盖，不再被启动脚本无条件覆盖。

### Worker 可观测性

文件：`Isaaclab_other/persistent_isaac_worker.py`

- 每次请求记录 slot、任务名、env 数、课程阶段和请求序号。
- 日志格式：`[WORKER] slot=... task=... num_envs=... stage=... request=...`。

### 之前已完成且保留的修复

- `evolution_tasks/task_branch_grasp/branch_grasp_env.py`：修复批量环境中局部碰撞端点与 branch 轴向向量的 batch 维度问题。
- `Isaaclab_other/main_evolution.py`：没有存活个体或没有可选子代时直接失败，不再伪造推进 generation。

## 3. 双 Worker 稳定性测试

实验名：`exp_20261001_parallel2_smoke`

测试配置：

- 2 个候选个体。
- 4 个任务：Grasp、BranchGrasp、Forage、Strike。
- 每个任务 3 次 PPO iteration。
- `ISAACLAB_NUM_ENVS=4096`。
- `EVOLUTION_PARALLEL_SLOTS=2`。
- `EVOLUTION_PARALLEL_SPLIT_ENVS=1`，即两个 worker 各 2048 env。
- `EVOLUTION_ISAAC_WORKER_MAX_REQUESTS=4`。
- persistent worker 开启。

实际结果：

- 两个 Isaac 进程同时运行，均使用 `cuda:0`，slot 0 和 slot 1 均成功收到请求。
- 每个 slot 连续处理 4 个任务请求，完成后正常退出。
- 两个 worker 没有崩溃、超时或自动重启。
- 没有遗留 `.request.json` 或 `.working.json` 文件。
- 运行状态正常推进到 generation 1，`parallel_slots=2`。
- 观测到 GPU 利用率约 80%，显存约 `17.5 / 49.1 GB`；两 worker 并发期间没有显存溢出。
- worker 日志显示单 worker 的 PPO rollout 吞吐约 4.3k--8.0k fps，说明测试确实完成了 Isaac 仿真而不是只启动进程。

## 4. 当前结论

短规模下，双 worker 并行已经验证可用。当前机制是：同一张 4090 上启动两个相互隔离的 Isaac 进程，每个进程使用一半环境数；场景初始化仍通过锁串行化，初始化完成后任务请求可以并发运行。

这不能直接保证 15 generation 长训练永不发生 native 层问题，但已经排除了“单卡双 worker 必然不能运行”的判断。正式训练建议保留：

```text
EVOLUTION_PARALLEL_SLOTS=2
EVOLUTION_PARALLEL_SPLIT_ENVS=1
ISAACLAB_NUM_ENVS=4096
EVOLUTION_ISAAC_WORKER_MAX_REQUESTS=4
EVOLUTION_REUSE_ISAAC_PROCESS=1
```

正式训练时重点监控：

- `Isaaclab_other/exp_*.launch.log`
- `parallel_eval_slots/slot_0/isaac_worker.log`
- `parallel_eval_slots/slot_1/isaac_worker.log`
- `nvidia-smi` 中显存是否持续低于 45 GB。

如果双 worker 在长训练中出现 native 崩溃，优先将基础环境数从 4096 降到 3072 或 2048，而不是立即关闭双 worker；这样仍可保留并发调度带来的收益。

## 5. 下次正式训练注意

- 训练参数必须在启动命令中显式写出，避免沿用旧实验默认值。
- 若使用 `stage1=150`、`stage2=400`、`stage2_top_fraction=0.25`，应在启动前检查日志中的实际值。
- 新实验使用新的 `EVOLUTION_EXPERIMENT_NAME`，不要覆盖旧实验目录。
- 先观察至少一个任务批次同时出现 `slot=0` 和 `slot=1` 的日志，再认为双 worker 已真正生效。

## 6. 2026-10-01 碰撞 Gate 调查与修改

正式实验 `exp_20261001_fast15_parallel2` 已暂停。问题不是主进程死锁，而是 generation 0 的候选形态筛选成本过高：约 56 分钟仍未进入 PPO，GPU 利用率常在 97%，但显存只有约 3 GB，说明资源被单个 Isaac 碰撞验证占用。

抽查已完成的 runtime 报告后，主要失败模式为：

1. 邻近长指的基节在活动范围扫掠时发生约 1.5--4.3 mm 的几何重叠，例如 `link_1_0` 与 `link_2_0`；这更像当前“全 ROM + 严格深度阈值”的筛选失败，不等于 mesh 损坏。
2. 个别变异导致运行时关节限位误差约 0.4--0.9 rad；这是兼容配置与变异后关节状态不一致，需要单独修复，不应与 mesh 畸形混为一谈。
3. 碰撞过滤器缺失或多余不是主因；已检查失败报告，`missing_filters` 和 `unexpected_filters` 基本为空。

已完成修改：

- `Isaaclab_other/collision_gate.py` 新增 `lightweight_geometry_prefilter()`，只在 CPU 上检查拓扑、尺寸、有限值、关节变换和安全范围。
- `Isaaclab_other/main_evolution.py` 在初始个体和变异子代进入 Isaac 前先运行该预筛选；不通过者不启动 Isaac。
- `run_local_evolution_4090_15g_single_stage_reuse.sh` 默认关闭高成本的 `EVOLUTION_SCRIPTED_PREFLIGHT`，仍保留显式设置为 `1` 的能力。
- 通过预筛选的候选仍进入原有 Isaac/PhysX runtime gate，因此没有取消最终物理安全检查。

当前没有训练或 Isaac 子进程运行。下次启动时应先观察候选筛选速度；若仍被 runtime gate 占满，再考虑将连续 ROM 检查改成分级策略，而不是直接放宽所有碰撞约束。

## 7. 轻量预筛选 smoke test

实验：`exp_20261001_prefilter_smoke`

- 2 个候选、单任务 Grasp、每任务 1 次 PPO iteration。
- 双 Worker，各 512 env；脚本预演关闭。
- 结果正常推进到 generation 1，没有 worker 崩溃或遗留请求。
- 两个 slot 同时完成首批 Grasp 请求，随后继续处理下一代候选。
- 本次候选没有触发预筛选拒绝，因此记录到 `0` 个 `[PREFILTER]`，启动了 4 个 PPO 请求。

结论：轻量预筛选已验证不会阻断正常候选，但在这一小批正常变异中没有显著减少 Isaac 启动次数。它主要能拦截数值非法、尺寸超界和拓扑错误；要进一步减少当前常见的动态指节重叠，需要新增 CPU 近似运动学/包围体筛选，不能仅靠当前静态尺寸检查。

## 8. 变异后的拓扑自适应调查

正式训练再次暂停后检查发现，碰撞过滤器确实会从生成后的 URDF 拓扑动态生成，增删 link 不会直接沿用旧的碰撞 pair。问题在于旧的几何变异函数只修改了 `geometry_length` 或 `geometry_radius`，没有同步子 link 的 joint origin。

典型例子：`link_2_0` 长度从 `0.045 m` 增长到 `0.054 m` 时，旧代码仍把 `link_2_1` 固定在 `z=0.050 m`；而 capsule 的实际远端连接位置应移动到 `z=0.059 m`。这会使子指节相对父指节发生重叠或脱节，随后被 runtime collision gate 淘汰。半径变化也有同样问题，因为生成 capsule 的端点距离为 `length + radius`。

已修改：

- `Isaaclab_other/tools.py` 新增 `_shift_direct_child_joint_origins()`。
- `change_link_length()`：同步直接子关节的局部 Z 连接位置。
- `change_link_radius()`：同步半径变化造成的远端连接位置变化。
- `change_finger_length()` / `change_thumb_length()`：对整根手指逐节缩放并逐节传播连接点变化。

真实 `exp_20261001_fast15_parallel2.json` 中的 `0_0` 手型已通过测试：

- 单节长度变异：`0.045 -> 0.054 m`，子关节 `0.050 -> 0.059 m`。
- 单节半径变异：`0.005 -> 0.006 m`，子关节 `0.050 -> 0.051 m`。
- 整指长度变异：各节均同步缩放，拓扑连接测试通过。

这说明此前“只要增长就容易碰撞”并非变异必然导致，而是变异没有完成运动学连接适配。旋转变异仍受当前解剖约束：屈伸链上的随机 RPY 被禁止，根部姿态由手掌/外展兼容层生成；这避免了手指向手背反折，但也意味着旋转探索不是完全自由的。

## 9. 2026-10-01 形态-运动学同步与拓扑自适应

按“变异同步修改形态和运动学连接，测试时验证碰撞”的方案完成修改：

- `tools.py` 新增 `synchronize_kinematic_connections()`，对同一手指相邻指节，将子关节局部 Z 位置自动重设为父指节的 `geometry_length + geometry_radius`，保留横向偏移。
- `variation.py` 在所有变异完成后统一调用同步函数，避免长度、半径、平移等不同路径产生不一致连接。
- `add_link()` 不再使用 UUID 名称，而是生成连续的 `link_<finger>_<segment>`，保证末端解析、URDF 轴约束和任务兼容层能够识别新增指节。
- `remove_link()` 禁止删除五根手指的根节，允许删除末端节，避免生成器失去必需的掌指根；删除末端后链号保持连续。
- `collision_gate.py` 的轻量预筛选新增指节连续性与父子远端连接检查，在启动 Isaac 前拦截已知不一致候选。

验证结果：

- CPU/URDF smoke：长度变异、半径变异、新增末端指节、删除末端指节均通过。
- Isaac/PhysX bilateral smoke：半径、新增、删除三类案例均通过右手/左手碰撞、关节配置和 49 个姿态样本检查。
- 长度案例的连接和 URDF 配置通过，但该具体样本在全姿态扫描中出现真实指间严重重叠，因此被正确淘汰；这属于个体物理不可行，不是拓扑适配失败。

正式训练应使用新的实验名从 generation 0 开始，保留旧实验结果，避免新代码与旧 lineage 混用。

## 10. 2026-10-02 版本管理落地

- 确认 `/home/zjh/Evolution_PC` 使用 Git 管理，远端为 `Jiahe-Zhao/Evolution-Hand`。
- 训练运行中的日志、checkpoint、lineage、runtime state 和碰撞缓存继续由 `.gitignore` 排除，不进入代码版本。
- 当前已验证可运行的形态自适应、碰撞缓存、CPU 预筛选、双 worker 调度和 `stage1=100/stage2=300` 配置纳入一次代码提交。
- 后续每次修改先在本文件增加简短变更记录，再进行语法检查或 smoke test；确认训练能够启动后提交 Git。

## 11. 2026-10-03 训练/测试场景一致性与指尖 IK BC 门控

- 评测入口新增 `--curriculum_stage`：自动 checkpoint 会从 lineage 读取 `stage1/stage2`，并在导入任务配置前设置对应环境变量；报告记录实际阶段和 `reset_dof_pos_noise`。
- Grasp 训练与脚本测试共用环境原生 reset。球体由中间三根手指的近端支撑点、手掌局部法向和形态自适应偏移计算；当前共享配置为 `proximal_support_clearance=0.045`、局部偏移 `(0.012, 0, 0.011)`。
- 脚本新增 `--training_scene`，该模式不重新放置或固定球体，只使用训练时的指尖 IK 动作接口；测试确认初始碰撞间隙为正且无关节越限。
- BC 数据必须同时满足：动作由指尖接口提交、没有脚本关节覆盖、没有场景外部写入、轨迹通过任务成功判定。无效轨迹由主流程和 worker 双重拒绝。
- 当前单形态脚本在真实训练场景中可产生拇指和长指接触，但尚未稳定完成有序 M1/M2/M3，因此本轮正式训练使用既有第 15 代策略权重热启动，不注入未经验证的 BC；BC 入口保留为后续验证成功轨迹自动启用。

## 12. 2026-10-04 BC 预检一致性修正

- 修正进化主流程：Grasp 脚本预检现在显式传入 `--training_scene`，不再使用与正式训练不同的重定位/固定球体场景。
- 指尖 IK replay 在接近阶段使用原生 reset 几何，闭合与保持阶段跟踪实际物体位置，避免球体发生微小位移后继续追踪旧锚点。
- 形态删除必要指节导致的配置正则匹配失败应视为运动学/形态 gate 失败，不能进入 BC 或 PPO。
- 旧的 `exp_20261003_fingertipIK_scenealigned_15g` 已停止；新的第 0 代训练必须使用新实验名，并从已验证父代 checkpoint 继承策略权重。
