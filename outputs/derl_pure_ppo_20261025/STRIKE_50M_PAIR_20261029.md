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
