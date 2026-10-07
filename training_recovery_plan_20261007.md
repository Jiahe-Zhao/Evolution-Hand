# 2026-10-07 四任务测试诊断与后续训练方案

## 已核实事实

- v9 已训练完 15 代。用户对 `13_14` 形态的 stage2、种子 7/8/9 测试为 Grasp 3/3、Branch 0/3、Forage 1/3、Strike 0/3。
- Branch v9 训练使用 `EVOLUTION_BRANCH_BC_MODE=1`：20 维关节目标动作、树枝 reset 偏移及中指 spread 初值；旧评测入口未恢复该模式，造成动作语义和场景同时失配。种子 Branch stage1 的 `behavior_cloning.json` 确认执行了 149 样本、10 epoch BC。
- 已修复 `evaluate_task.py`：在构建环境前读取所选 checkpoint 的 `params/policy_contract.json`，自动设置 Branch 控制模式，拒绝缺失、跨任务或未知契约，报告记录 controller。相同 checkpoint、形态和种子 7/8/9 复测 Branch 3/3 成功，均在 82 步达到 stage2 的 15 步稳定接触；结果在 `outputs/eval_v9_branch_contract_fixed/branch/`。因此当前 Branch 0/3 主要是评测错误，不应据此重训。
- Strike 三次均在 38 步终止，各得 250 分（抓工具奖励），击打接触力 0。现有脚本的抓握阶段写 `scripted_joint_target`，提交的手指动作近乎为零，不能当作 BC；v9 预检没有一条 Strike 全任务成功轨迹。
- Forage 三次有一次 stage2 完整成功。v9 两条 Forage 脚本通过轨迹都属于 stage1，其清叶阈值是 0.040 m；stage2 阈值为 0.075 m。因此这两条不足以直接标记为 stage2 成功 BC。现有任务预算 Forage/Strike stage1 25、stage2 50 次 PPO 迭代，明显短于 Grasp/Branch 的 100/250。

## 执行顺序

1. **统一评测契约。** 所有任务的评测报告记录 checkpoint、控制器、课程阶段、reset 噪声、场景参数、形态和代码版本。读取契约后才创建环境；动作维度、控制器或场景签名不匹配时直接失败。对固定形态至少 20 个未参与示教的种子复测，并把每个任务的接触/阶段指标与全任务成功分开报告。Branch 先用已修复入口建立新基线；Grasp 使用同一入口回归。
2. **Strike 优先做完整示教。** 先记录 38 步提前终止的具体条件（工具高度/横向越界、失手、腕部命令及目标距离）。在原生 stage2 reset 中构建“抓稳工具→保持接触→腕部移动→撞击目标”的 23 维策略动作轨迹。所有动作必须走 `env.step(action)`；不得用 `scripted_joint_target`、直接写工具位姿或改变成功阈值。只有真实击打触发终局 1000 分、动作目标在关节限位内、物理瞬时超限不超过记录的 0.02 rad 容差、动作/观测有限且场景未改写的轨迹才输入 Strike BC。BC 后继续 PPO；对比现有 25/50 与适度增加 Strike 专项训练预算的效果。
3. **Forage 先补 stage2 数据，再做 BC 对照。** 从原生 stage2 场景收集两片叶子都达到 0.075 m 门槛的完整轨迹，可从脚本改进或现有成功策略的 rollout 导出逐步观测与真实 23 维动作。覆盖不同 reset 种子和叶子接触顺序，避免只有单一成功样本。以数据质量门控接入同任务 BC，然后与“只延长 Forage PPO 预算”做同预算对照。若 BC 在未见过的种子上无增益，则保留 PPO 方案，避免把偶然成功过拟合为示教。
4. **分层验收。** 同一形态、相同 stage2 参数和固定未见种子，对旧 checkpoint、修正评测后的 v9、Strike/Forage 新模型分别评估；先报告 20 个以上回合的成功数与置信区间，再扩展到多个形态。Strike 同时报告抓稳率、越界率、命中接触力和全任务成功率；Forage 分别报告两片叶子的清除率及全任务成功率。视频只保留具有物理成功证据的示教与典型失败案例。新模型必须在完整任务成功率上优于一致评测的基线，才替换旧 checkpoint。

## 约束与产物

- 不修改手的 URDF、网格、关节结构或限位；如确需修改，先请用户确认。
- 每条 BC 轨迹保存场景/动作契约、课程阶段、形态、种子、逐步观测与动作、接触/成功证据。失败或脚本关节覆盖轨迹不得进入 BC。
- 交付评测与示教代码、质量门控、可复现命令、成功/失败视频、对照结果表、MD 记录及相关 Git 提交。无需重跑已修复且 3/3 成功的 Branch 训练。

## 2026-10-08 执行进度

- Forage 原 v9 stage2 策略在种子 100–119 上全任务成功 14/20；另一组留出种子 200–219 上为 11/20。原用户三种子测试 1/3 反映了明显的种子波动，不能单独判定整体成功率。
- 从八条原生 stage2 全任务成功的 Forage 策略轨迹提取了 1526 组真实观测与 23 维动作，均经过成功事件和动作来源门控。以相同起始 checkpoint、形态 13_14、种子 17 和 100 PPO 迭代做 BC+PPO 与纯 PPO 对照。留出种子 200–219 的结果为：原 v9 11/20，BC+PPO 4/20，纯 PPO 12/20。BC 明显有害；纯 PPO 比原策略多成功一次，证据不足以替换原 checkpoint。按用户决定停止 Forage BC 路线，保留原策略和 PPO 训练结果供后续使用。
- Strike v9 只到达抓握奖励，缺少有效击打。在 stage2 原生场景中，以拇指初始展开 -2.0 rad、预抓握动作 0.55 的组合，原脚本得到一次物理全任务成功：138 步、1250 分、冲击力 115.5 N、目标横向误差 0.0332 m、同时保持拇指及四指接触。该原脚本使用关节目标覆盖，未进入 BC。
- 新 Strike 控制模式把 20 个真实关节（含展开关节）映射到 23 维 `env.step(action)` 的前 20 维，后三维为腕部动作；策略契约保存控制器和 reset 参数，评测入口在创建场景前恢复。原生 stage2 上找到两条不同动作且全任务成功的轨迹：闭合比例 0.78 和 0.82，环指展开目标均为 -0.08 rad。两者均在 138 步完成、无超过 0.02 rad 容差的关节瞬时超限，实测最大分别为 0.0104 和 0.0184 rad。全部动作目标保持在硬限位内。质量门控合并 276 样本，BC 训练 10 epoch 最终损失 0.192，随后以 4096 环境、horizon 16、minibatch 4096、mini epochs 5、seed 17 启动 100 轮 PPO。
- 同一脚本种子 7–11 的五条轨迹逐字节相同，表明此 stage2 reset 对这几个种子没有提供场景多样性；因此验收不能把 5/5 当作独立泛化证据。后续应明确评估 reset 扰动和形态迁移。
- 合格 Strike 示教视频：`outputs/strike_bc_direct/validation/strike_success_bc.mp4`；本地副本为 `/Users/zhaojiahe/Documents/科研/strike_bc_success_13_14.mp4`。
- 首次两风格 276 样本 BC（10 epoch，loss 0.192）加 100 轮 PPO 后，独立评测只得到 250 分抓握奖励，35 步掉落工具；没有完整击打。单风格 138 样本 BC 拟合 200 epoch 后 loss 0.000192，但 PPO 前闭环仍在第 34 步掉落工具。三轨迹 415 样本 BC 拟合 loss 0.000176，闭环延长至 50 步，仍未击打。示教拟合误差小不等于闭环成功。
- 对真实动作加入固定种子的低幅度相关扰动，逐条运行原生 stage2 物理验证。0.005 标准差的 10 次中仅 2 次合格；0.002 标准差的 30 次中 7 次合格。其余因任务失败或超过 0.02 rad 瞬时关节超限被排除。原示教加 9 条合格扰动轨迹构成 10 条不同动作、1390 样本的数据集 `outputs/strike_bc_direct/strike_bc_10diverse.npz`。
- 10 轨迹 BC 拟合 200 epoch，loss 0.0000547。PPO 前独立评测 `outputs/strike_bc_direct/eval_10diverse_bcinit/strike/evaluation.json` 在 19 步得到 1250 分，其中稀疏击打奖励 1000 分；接触力 32.58 N、目标距离 0.02053 m、拇指接触 27.65 N、四个长指接触、连续抓握 19 步。已保存 `evolution_tasks/logs/evolution_task/strike_bc_stage2_13_14_10diverse_seed17_bcinit/nn/bc_init.pth`，再从该成功 checkpoint 继续 100 轮 PPO。

## 2026-10-08 最终对照与模型选择

| 模型 | stage2 种子 200–219 全任务成功 | 单回合典型结果 | 结论 |
| --- | ---: | --- | --- |
| Strike 10 轨迹 BC 初始模型 | 20/20 | 19 步、1250 分、32.58 N 有效接触 | 保留为当前固定形态推荐模型 |
| 由上述 BC 模型继续 100 轮 PPO 的终点 | 0/20 | 599 步、250 分抓握奖励、无有效击打 | PPO 退化，不替换 BC |
| 同轮 PPO 按训练回报保存的最佳模型 | 单回合 0/1 | 52 步、250 分 | 不推荐 |

20 个种子的初始几何完全相同，每个模型的 20 条测试轨迹也逐字节一致；因此表中的 20/20 和 0/20 只是相同 reset 的重复性验证，不是跨场景泛化成功率，也不适合计算独立二项置信区间。PPO 终点曾记录到 398 N 接触力，但当帧拇指接触为 0、仅一个长指接触、抓握连续步数为 0，且目标距离 0.060 m；严格成功判定正确地拒绝该碰撞。

旧 v9 Strike 策略在新 reset 参数下的参考单回合为 0/1、453 步、250 分。旧模型训练时使用不同 reset 参数，因此此结果仅说明它不能直接替代当前策略，不作为严格同分布训练对照。

跨形态诊断：同一 BC checkpoint 在另一已有形态 `13_13` 的 stage2 单回合为 0/1、87 步、250 分；关节顺序契约通过，但仅完成抓握。当前推荐仅适用于训练形态 `13_14`，尚不具备跨形态可迁移性。该测试只生成输出目录中的临时形态资产，没有修改源手结构文件。

推荐 checkpoint：`evolution_tasks/logs/evolution_task/strike_bc_stage2_13_14_10diverse_seed17_bcinit/nn/bc_init.pth`。100 轮 PPO checkpoint 作为实验结果保留；不改写 v9 的全局 lineage 选择。下一步若要提升泛化，应先在训练和测试中共同引入可控的 reset 扰动，并用闭环纠错示教或 BC 约束 PPO，避免当前纯 PPO 将完整击打退化成抓握。此项会改变场景分布，需单独记录版本与对照。

成功脚本视频：`outputs/strike_bc_direct/validation/strike_success_bc.mp4`；成功 BC 策略视频：`outputs/strike_bc_direct/eval_bc_policy_video/strike/episodes/episode_000/successful_videos/episode_000_seed_200_success_step_18.mp4`。本地可查看 `/Users/zhaojiahe/Documents/科研/strike_bc_success_13_14.mp4` 和 `/Users/zhaojiahe/Documents/科研/strike_bc_policy_success_13_14_slow.mp4`（后者仅把播放速度放慢四倍，帧内容未改）。
- 本次未修改手的 URDF、网格、关节结构或限位。

## 2026-10-08：完整四任务代际训练准入复核

本次复核区分单形态可复现成功与可启动跨形态 15 代训练。`15_0` 的旧脚本目录有 Grasp、Branch、Forage、Strike 各自成功记录，其中 Grasp `grasp_margin.json` 的网格穿透步数为 0；但旧 Strike 成功轨迹使用关节目标覆盖，不能作为策略 BC。新 Strike 10 轨迹 BC 在 `13_14` 原生 stage2 场景固定几何下达到 20/20，`13_13` 单回合为 0/1；PPO 后为 0/20。它证明固定形态动作可执行，但尚不能证明跨形态或随机 reset 的完整击打能力。

当前 `main_evolution.py` 的第 0 代 BC 数据接入仅覆盖 Grasp/Branch，Strike 不在 `bc_eligible` 内；旧 `run_local_evolution_4090_15g_fast_multitask.sh` 还默认从旧场景 checkpoint 继承，并允许脚本预检失败后继续纯 PPO。按“Strike BC 可用且四任务通过训练门槛再启动”的条件，本轮**没有启动新的 15 代训练**。下一步需要给 Strike 添加经过逐形态、逐阶段闭环物理验证的原生策略动作示教及代际 BC 接入，并在与训练相同的 reset 分布中复测；PPO 阶段须保护成功 BC 行为，不能只以 250 分抓握奖励替代击打成功。Forage 保持已验证的纯 PPO 路线。

相邻形态 `13_13` 的 Strike 原生 stage2 脚本补测：沿用 `13_14` 的 0.78 闭合、环指展开 -0.08 rad 时，320 步未成功、奖励 500、260 步物理超限，最大超限 0.5636 rad（中/环/小指展开关节）；改为保守的 0.65 闭合、环指展开 0 时，仍未成功、奖励 250、245 步超限，最大 0.3301 rad。两条都不得进入 BC。没有修改手结构文件。记录在 `outputs/strike_bc_direct/validation/strike_13_13*.json`。

`13_14` 原生 stage2 的工具初始位置 ±2 mm x 偏移筛选（未改变成功阈值或结构）：基准闭合 0.78 在 +2 mm 时 102 步终止、0 分、无超限，在 -2 mm 时 320 步、0 分、247 步超限、最大 0.0977 rad。+2 mm 闭合 0.82/0.86 都在 102 步终止且 0 分；-2 mm 闭合 0.70 为 320 步 0 分且无超限，0.74 为 320 步 0 分且超限 245 步。所有轨迹都未达到全任务成功，不得加入 BC。结论是固定关节目标与单一工具几何耦合，下一步要基于工具位置及指尖接触做闭环预抓握；用相同 reset 扰动同时训练和测试，不能把固定场景的 20/20 当成扰动泛化。原始结果在 `outputs/strike_bc_direct/validation/strike_offset_*.json`。

偏移失效机理复核：基准 `13_14` 轨迹从第 0 步到命中一直保持拇指及四长指接触；+2 mm x 偏移在第 10 步已失去拇指接触，随后工具飞离；-2 mm 偏移从第 5 步开始拇指力接近零。仅改变 reset 拇指展开角（-1.8/-1.9/-2.1/-2.2 rad）没有完整成功。试验性拇指/长指不同闭合速度保留基准 1250 分成功，但两个偏移方向仍失败；该无效脚本改动已撤回。下一步应引入工具位置和接触力反馈的预抓握纠错，而不是继续扫固定闭合参数。原始诊断为 `outputs/strike_bc_direct/validation/strike_rate_*.json`。

## 2026-10-08：第 0 代 Strike BC 接入与共同种子预检

`main_evolution.py` 已把 `EVOLUTION_STRIKE_BC_MODE=1` 纳入 stage1 预检与第 0 代 BC：Strike 使用 162 维动作前观测、23 维真实策略动作，拒绝脚本关节覆盖、场景改写、非有限值、未完成击打或超过 0.02 rad 的物理关节超限；旧笛卡尔 Strike checkpoint 不再错误继承到新 direct-joint 控制模式。预检缓存签名包含 BC 模式、reset 和脚本参数，防止不同场景复用旧成功报告。`13_14` 的原生 stage1 Strike 轨迹复测为 138 步、1250 分、全任务成功、0 步超限、最大瞬时超限 0.0104 rad，满足该 BC 门槛。

共同种子预检仍未通过：`13_14` 的同 stage1 训练场景中，Grasp 脚本持续接触且网格穿透 0，但完整成功判据为 false；Branch 原生动作脚本 false 且 5 步关节超限；Forage 脚本 false。策略评测的 Grasp/Branch/Forage 成功率是另一项证据，不能替代此处的第 0 代脚本预检。`15_0` 的 Strike 原生脚本完成抓握和物理击打（奖励 1500，28.07 N），但环指/小指展开关节有 77 步超限，最大 0.1366 rad，因此不可入 BC；环指目标 -0.24、-0.16、0、0.10、0.20 rad 的有限筛选均未获得合格全任务轨迹。以上没有修改手结构文件。原始记录在 `outputs/preflight_13_14_four_task/`。

故当前没有四任务共同合格的第 0 代形态，**未启动**新的 15 代训练。Forage 按既定决定继续采用 PPO，不加 BC。下一步优先解决 `15_0` Strike 受力造成的展开关节物理超限，或让 `13_14` 的 Grasp/Branch 脚本达到各自准入门槛；启动时必须使用和测试一致的 reset、原 PPO 参数，并保存独立 lineage。

### 2026-10-08：历史共同种子筛选

在 v9 的 120 个 preflight 目录中，只有 `15_3` 和 `1_4` 的 Grasp、Branch、Forage 三任务同时显示成功。对这两个形态使用新 Strike 23 维原生动作、stage1、相同 reset 重新验证：`15_3` 320 步 0 分、无超限，未建立抓握；`1_4` 完成抓握和真实击打，1500 分，但中/环/小指展开关节超限 102 步，最大 0.3240 rad，不得进入 BC。`1_4` 闭合比例 0.70、0.74、0.76 的有限试验均未改善。将 `15_0` 的 Strike actuator damping 从 0.2 提到 1.0 也使击打退化到 500 分、超限增至 0.5376 rad，诊断性阻尼改动已撤回。`main_evolution.py` 本身现纳入 preflight 缓存签名，修改预检调用逻辑时会强制重新验证。上述历史三任务结果需要在新入口下复测才可作为新训练准入；目前没有四任务共同合格候选。

`15_0` Strike 腕部击打速度诊断：每步下降 0.25 mm 时物理击打仍发生，但超限 194 步、峰值 0.1368 rad；1 mm 时击打仍发生、超限 20 步、峰值 0.1000 rad；2 mm 为 31 步/0.1834 rad，4 mm 为 25 步/0.2113 rad。都超过 0.02 rad，且更快速度不能保证稳定单次击打。速度参数的试验性代码已撤回，轨迹未入 BC；记录在 `outputs/strike_bc_seed_search/seed15_*`。

### 2026-10-08：13_14 同场景 stage1 策略准入对照

使用固定形态 `13_14`、原生 stage1 环境、未见种子 200–209，逐回合完整任务成功事件复测：Grasp v9 stage1 checkpoint 10/10；Branch v9 stage1 direct-joint checkpoint 10/10；Forage v9 stage1 checkpoint 1/10，但同一形态的纯 PPO stage2 checkpoint 在同一 stage1 场景为 9/10；Strike 10 条合格示教的 BC 初始 checkpoint 在同一 stage1 场景为 10/10。结果在 `outputs/admission_13_14_stage1_policy/`。四个所选 checkpoint 文件均存在，策略契约匹配：Grasp/Forage 使用 Cartesian 控制，Branch 使用 `branch_joint_target_v1`，Strike 使用 `strike_joint_target_v1`，Strike 契约包含拇指 reset -2.0 rad、预抓握 0.55、工具偏移 0。Forage 采用纯 PPO 策略，不加 BC。

这组对照证明已有策略在当前 stage1 测试场景的表现，但不自动使失败的 Grasp/Branch/Forage 脚本预检通过。Strike reset 几何对这些种子固定，其 10/10 是重复可执行性而非独立泛化。原项目说明中的“共同形态四任务脚本验证”门槛和“可用同场景成功策略评测替代部分脚本”的解释目前存在分歧，已向用户请求准入标准澄清；在此之前没有启动新的 15 代训练，也没有修改手结构文件。

同一 `13_14` 形态 stage2 补测（种子 200–209）：Grasp v9 stage2 checkpoint 10/10，Branch v9 stage2 direct-joint checkpoint 9/10，记录在 `outputs/admission_13_14_stage2_policy/`。已有 Forage v9 stage2 在种子 200–219 为 11/20、在 100–119 为 14/20；Strike BC 初始模型在 200–219 为 20/20，但 reset 几何只有一种，不视为独立扰动泛化。至此四任务 stage1/stage2 的固定形态策略证据齐备；共同脚本预检仍未全部通过，等待准入标准澄清。

### 2026-10-08：Strike BC 初始化稳定性与 PPO 遗忘

隔离试验保留原 PPO 参数（4096 环境、horizon 16、minibatch 4096、mini epochs 5）。同一 10 轨迹/1390 样本数据，重新随机拟合 200 epoch 的损失 0.0000655，但 PPO 前闭环只有 250 分；400 epoch 损失进一步降到 0.0000229，PPO 前仍为 500 分、未击打。低监督损失不能替代物理闭环验收。原已验证 BC checkpoint 在固定 reset 上 stage1 10/10、stage2 20/20；映射到隔离训练 slot 后权重逐项完全相同。以它起步，一轮 PPO 后 0/1、250 分；每轮重放 5 批或 50 批示教也未恢复完整击打，95% actor 锚定乃至仅保留约 0.1% actor 更新的离线诊断仍失败。上述 BC replay/锚定试验代码已撤回，输出保存在 `outputs/strike_bc_replay_smoke/`。

进一步用原生脚本测试 `13_14` 工具 reset 在 x/y/z 方向各 ±0.2 mm：六条均未通过既定 BC 门槛；部分物理击打成功但关节超限 0.053–1.800 rad，全部排除，见 `outputs/strike_bc_reset_diversity/`。目前 Strike BC 仅能保证固定 reset 的已验证 checkpoint，不具备小扰动泛化，不能用这六条扩充训练数据。训练 worker 新增受保护路径：当 Strike 从已验证 checkpoint 继承且 `EVOLUTION_BC_EPOCHS=0` 时，要求关节形态和 reset 契约完全一致、跳过随机重拟合，并在 PPO 前保存 `bc_init.pth`；该文件须单独闭环验证且不得因 PPO 退化被覆盖。新一轮四任务训练仍未启动。

### 2026-10-08：成功策略动作导出、原样回放与物理审计

对固定形态 `13_14`、stage1 训练场景、种子 200，使用评测器现有 `--export_success_bc` 导出 Grasp/Branch/Forage 成功回合的完整逐步动作；新增 `--replay_policy_trace` 在相同种子、形态及 checkpoint 控制契约下逐步提交原动作，并审计物理关节硬限，Grasp 额外用生成的源网格审计球体穿透。动作回放不是新训练策略，也没有修改手结构或场景。审计输出在 `outputs/admission_13_14_replay_policy_traces_audited/`，原轨迹在 `outputs/admission_13_14_exported_policy_traces/`。

- Grasp：46 步完整成功，M3 拇指加两长指连续 10 步；源网格穿透 0 步，最小间隙 0.709 mm；关节超限 0。
- Branch：75 步完整成功，关节峰值超限 0.01725 rad，超过 0.02 rad 的步数 0。
- Forage：98 步完整成功，但第 96 步关节超限 0.5572 rad，因此这条动作不能作为合格形态脚本或 BC。

继续筛选 Forage 同场景已成功的种子 202–209：8 条均可原样回放完成全任务，但每条都有超过 0.02 rad 的物理硬限超越；峰值分别为 1.4373、2.5091、0.1710、1.0373、0.0497、0.1504、0.0514、0.0315 rad（按 202、203、204、205、206、207、208、209 顺序）。输出在 `outputs/forage_quality_seed_*_{export,replay}/`。这不影响先前纯 PPO 成功率的统计，却说明成功事件不足以作为严格物理质量证明。没有将这些轨迹加入 BC，仍未启动新 15 代训练。Grasp/Branch 的成功轨迹可以作为固定形态示教候选；下一步须使 Forage 在原场景成功且不超物理硬限，并解决 Strike BC 在 PPO 后遗忘的问题。

Forage 扩展筛选种子 210–219：10/10 完整成功，但按同一 0.02 rad 物理超限阈值均不合格，峰值 0.0305–1.2834 rad；记录在 `outputs/forage_quality_seed_search/`。种子 209 的较轻超限发生在第 85、95 步环指展开关节，实际角度约 -0.201 rad、物理下限 -0.170 rad，而控制目标仍在限位内（约 -0.150 rad），说明是实际碰撞动态使关节越限，不能仅裁剪动作解决。Forage 的纯 PPO 完整成功率与物理质量需要分开报告；若坚持 0.02 rad 门槛，当前没有合格的 Forage 固定形态示教。尚未修改结构、关节限位或任务物理参数。

### 2026-10-08：Strike 首轮 PPO 遗忘定位与阶段 BC 保护

比较同一 `13_14`、stage2、种子 200：已验证 BC 与仅保留 0.1% PPO actor 更新的模型在首步动作最大相差 0.000892，但下一步物理观测已分叉；原 BC 重跑两次的观测和动作逐步完全一致。用原 BC 成功动作只保护前 1 或 5 步仍失败；保护前 10、15 或 18 步后交给 PPO，三种均完整击打且无超过 0.02 rad 的关节超限。接触里程碑在第 9 步出现，说明之前 PPO 退化主要起于抓握接触建立阶段。

新增可选冻结 BC 教师：Strike 使用原 162→23 动作 BC checkpoint，在 `tool_was_held` 为 false 时以原 checkpoint 的观测裁剪、归一化及 ELU actor 控制手指；一旦实际工具接触里程碑成立，后续动作由 PPO 输出。教师在环境动作入口生效，因此训练和评测相同；教师 checkpoint 的关节形态、reset 契约和 SHA256 进入策略契约，评测时自动恢复。评测器禁止把阶段覆盖下的 PPO 提交动作标记为成功 BC 示教，并保存实际执行动作供审计。未修改手结构文件。

隔离冒烟训练使用原 PPO 设置（4096 环境、horizon 16、minibatch 4096、mini epochs 5），从已验证的 `13_14` BC checkpoint 继承，`EVOLUTION_BC_EPOCHS=0`，运行 1 轮，训练均分 256.84976。输出 `outputs/strike_bc_replay_smoke/strike_phase_teacher_1iter_v5_strike_p_EvolutionHand-Strike_stage2/`。该训练后的 checkpoint 在 stage2 未见种子 200–219 完整成功 20/20，20/20 无超过 0.02 rad 的关节超限；stage1 未见种子 200–209 自动按契约恢复教师后完整成功 10/10，10/10 无越限。证据在 `outputs/strike_bc_replay_smoke/eval_phase_teacher_trained_20/` 和 `eval_phase_teacher_stage1_auto_10/`。这证实固定形态下阶段 BC 保护能避免首轮 PPO 遗忘；跨形态与随机工具 reset 尚未验证，因此此处只是隔离训练试验，未启动新 15 代正式训练。

Forage 纯 PPO 的完整成功率仍成立，但按 0.02 rad 严格物理门槛，其已检查的 19 条成功轨迹都超限；若保留该门槛，需要在不修改手结构的前提下找到合格轨迹或调整训练与测试共同使用的控制/场景。四任务正式准入仍未全部满足。
