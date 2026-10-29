# Strike 5000 万步/手：先机械手，再 Shadow

用户已明确授权先启动我们的机械手，Shadow 完成物理腕部适配及验证后接续。预算解释为每只手 50,000,000 步，共 100,000,000 步；256×128 整 rollout 实际为每只手 49,971,200 步（以 effective_config.json 为准）。

## 当前启动

主机 zjh@100.81.86.51，仓库 /home/zjh/Evolution_PC，代码 outputs/derl_pure_ppo_20261025。
启动脚本 launch_strike_50m_20261029.py，结果 strike_50m_pair_20261029/status.json。
32 环境、4096 步短 PPO 和三组完整回合评估已完成；随后自动启动 ours，256 环境长训。短测通过表示可运行、状态有限、评估闭环成立，不表示任务已学会。

机械手源：outputs/discrete_hand_evolution_20261024/checks/baseline/right/urdf/current_agent.urdf。保持已有物理 XYZ 腕部驱动、自由工具、真实支撑台和 open_grasp_v4 场景。无 BC、无父策略。全手触觉开启。
PPO：网络 64×64 tanh，可学习 Gaussian 方差，rollout128，初始学习率3e-4线性下降，gamma .999，4 epochs，minibatch512；每约210万步完整回合评估。训练seed2029、评估场景seed1062902。

## 必须区分的奖励语义

旧 CONTACT_CURRICULUM 默认 stage1 只完成抓握抬起，不能冒充 Strike。新任务采用 StrikeProgressAdapter，保留原生当前持工具击打事件 >=999 为唯一成功，抓握/抬起不会成功终止。采用与 Branch/Forage 同类型的 episode maximum 有界进展奖励，总上限10，击打成功奖励10。历史最大值进入观测；软接触力和时间代价，不因轻微动作变化增加失败门槛。此奖励是显式新适配，不是原样复用旧 stage1。

## Shadow 尚未就绪，不能说已排入可执行队列

- Branch 已有成熟对比接口：branch_comparison.py / branch_comparison_protocol.py，可参考 Shadow 关节、指尖和触觉语义映射。
- 旧 Shadow Strike 文件在 evolution_tasks/task_strike/shadowhand_strike_env.py、shadowhand_strike_env_cfg.py，使用旧圆锥/不同目标/旧奖励，不能直接当同场景比较。
- 必须保留原 Shadow 手指、掌部、碰撞几何和腱约束；接入与我们同等可达范围、速度和力限的物理 XYZ 外部腕部，避免直接改写固定基座位姿。
- 我们现有腕部适配：Isaaclab_other/driven_wrist_adapter.py、diagnostic_wrist.py；joint_graph_env.py 位于 outputs/trusted_hand_baseline_20261021/four_tasks。
- Shadow 需要同一自由工具、支撑台、击打目标、成功阈值、共同奖励、全手触觉，观测语义和 PPO 预算一致；记录固有手型/关节数量/驱动差异。
- 先做初始碰撞检查、静态保持/物理移动腕部检查、完整回合与短 PPO，再接续50m。不可把旧 Shadow 环境直接塞入队列，不可未经验证宣称公平对比。
- 长训 ours 正在占 GPU，不得为了适配停止或覆写其源快照/检查点。Shadow 长训应在 ours 完成后启动；相关文件、短测证据和启动PID写MD/git。若要修改手指/掌部结构，先询问用户。

## 启动确认

ours 长训 PID 1374695（timeout wrapper），首批确认迭代9、294912步、4345.8 steps/s，成功数暂0。早期峰值接触力469N、工具速度7.59m/s，仍需观察，不能把短测可运行说成接触已完全稳定。预计本轮约3.5–5小时，依后续速度及评估耗时更新。
已创建当前聊天自动接续 `strike`（每30分钟），继续Shadow适配、验证和后续启动；重要进展/异常/完成才通知。Shadow当前仍是pending，不是已启动或已完成适配。

## 自动接续：Shadow XYZ 独立物理验证完成；ours 出现严重异常

新增 shadow_xyz_wrapper.py / shadow_xyz_probe.py / test_shadow_xyz.py。Shadow 官方 USD 独立组合副本 shadow_xyz_20261029/shadow_xyz.usda：只把原 rootJoint 的 body1 改接 diagnostic_world，并增加 XYZ 驱动链；逐属性与关系检查原手指/掌部/腱等保持不变（排除原 rootJoint 的唯一重挂接）。原资产没有覆盖。新增移动质量0.1kg，±0.25m、0.02m/s、100N、刚度3000、阻尼100，与我们的外部腕部参数对应。

静态保持、local XYZ、world XYZ 共7段验证通过：1cm位移最大误差2.92e-6m，峰值刚体速度0.00820m/s，有限状态。首次测试把安装坐标系误当世界轴，失败报告已保留；正确转换后通过。
**集成必须使用 reset 时 robot0_hand_mount 的世界四元数，把 world offset 逆旋转到 stage 坐标；不能只用 cfg.init_state.rot。** Shadow 资产安装根有额外旋转。wrist_test.json 保存参考四元数与实际关节列表。

还未完成：共同 Strike 观测/原生成功接口、目标和工具/支撑位置对齐、初始碰撞、接触静态保持、完整回合及短 PPO。不得因无负载腕部通过就启动 Shadow 长训。

ours 本次初查11,862,016步、5次评估0成功；历史200.087m/s峰值首次在3,440,640步记录。后查27,951,104步、13次评估全0、历史峰值接触力1861N；速度峰值是历史累计，并不表示每一步都为200m/s。已通过异步问题请求用户暂停排查，尚未获得回复；训练未停止，不重复启动。应优先确认物理异常（TGS手部velocity iterations>4警告、接触/求解器/状态一致性）而非继续把零成功归因为训练步数不足。不得静默修改运行中模型或宣称已收敛。

## 2026-10-29 11:20 接续：ours 完成，Shadow 共同配置检查通过

ours 自然完成，exit_code=0，49,971,200 步，训练累计成功回合485。最终完整回合复评：随机动作种子1062903为52/256（20.3125%），1062904为30/256（11.71875%）；均值动作为3/256（1.171875%）。出现真实判据下的成功，但成功率低且随机策略差异明显，不能宣称稳定收敛。

历史累计最大工具速度200.08696m/s、最大接触力4400.473N，最大关节越限3.6597e-5rad。速度/力峰值不代表每个回合异常，也不能证明所有成功无效；需要逐成功回合物理复核。此前暂停问题已因训练自然完成失去必要性，未主动停止或重复启动。

新增 strike_shared_scene.py：从本轮实际 native_physics_config.json 提取并校验共享场景契约，重建工具和目标的配置；定义当前持握且工具对目标的过滤冲击力满足阈值的成功判据，以及 Shadow 拇指优先的五指刚体接触分组。shared_strike_contract.json 保留来源文件SHA256。

check_strike_shared_scene.py 已在主机 Isaac 环境通过12项检查：工具/目标的尺寸、刚体、碰撞、质量、材料配置一致；当前持握与过滤冲击判据正确；五指接触组完整且不重叠。证据 shared_scene_checks.json。**此检查仅验证配置和判据，不是完整场景仿真或短PPO通过。**

Shadow 仍未启动50M，也未排入可执行队列。剩余：加载共同场景，校准安装位姿及工具支撑位置，初始碰撞/接触静态保持/有负载腕部、全手触觉与动作观测接入、完整回合及短PPO。保留原手结构和腱约束。
