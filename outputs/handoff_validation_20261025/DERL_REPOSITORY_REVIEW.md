# DERL 官方仓库核对与本项目参考点

日期：2026-10-25。仓库：https://github.com/agrimgupta92/derl

审查提交：`91e077e355b27bb74fa920c708d606ff9f4d611e`。本地仅阅读源码，没有安装或运行 DERL；下列 DERL 数值来自实现或论文，不是本机复现结果。不要把 README 的 `ft_test.yml` 冒烟配置当论文训练配置。

## 1. 训练和进化的实际实现

`tools/evo_single_proc.py::tournament_evolution/ppo_train` 为子代创建新 PPO 对象，仅传入子代 XML。父代 metadata 用于 lineage 记录，没有加载父代模型；`PPO.__init__` 新建 ActorCritic。因此 DERL **继承结构，不继承已学习策略，也没有 BC**。

外层采用异步锦标赛：最近 576 个体作为活跃池，随机抽 4 个体（代码 random.choices，允许重复），按配置中的目标选父代，变异后训练子代。4000 是包括初始种群的累计评估量，不是一个同时存在的 4000 个体种群，也不是 4000 代。老化依赖 metadata 时间戳；旧模型可保留，但不再参与活跃池。父代选择支持 Pareto 多目标；ft.yml 只用前进奖励，mvt.yml 用操作奖励。

这与我们的“仅第 0 代 BC、后代弱遗传策略”不同。我们的方案可以节约探索成本，但适应度会混入父策略对新结构的匹配程度。要证明结构本身更优，最终候选需要相同训练预算、多 seed，以及统一初始化/从头训练的小规模复核。

源码：[进化工作进程](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/tools/evo_single_proc.py)、[选择逻辑](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/utils/evo.py)。

## 2. PPO 对比

| 项目 | DERL 正式默认 / ft.yml | 我们上一轮 8 个体 | 本次接续验证 |
|---|---|---|---|
| 起点 | 每个形态随机初始化 | 继承第 0 代 BC actor | 同一 BC actor，新增观测列零初始化 |
| Actor/Critic | 独立编码器；本体感知 MLP 64,64，Tanh | 独立 512,512,256,128，ELU | 沿用网络 |
| 采样 | 32 环境 × 128 步 = 4096 | 256 × 16 = 4096 | 32 × 16 = 512 |
| 优化 | 4 遍，每 minibatch 512 | 5 遍；以实际 YAML 为准 | 5 遍，每 minibatch 512 |
| 初始 LR | 3e-4，两网络相同，线性衰减 | Actor 1e-7 自适应；预热后 Critic 也降至 Actor LR | 对照沿用；校准组 Actor 1e-7、Critic 3e-4 |
| gamma / GAE lambda | 0.99 / 0.95 | 0.9995 / 0.95 | 同上一轮 |
| PPO clip / 梯度裁剪 | 0.2 / 0.5 | 0.2 / 1 | 同上一轮 |
| value coefficient | 0.5 | 4 | 4 |
| 探索 | log_std 可训练，初始 std=1；entropy coefficient=0.01 | 固定 std：手指0.0005、腕0.0001；entropy=0 | 本次保留，以隔离接续机制 |
| 归一化 | 本体观测运行均值方差；奖励按折扣回报标准差缩放 | 固定物理缩放；value normalization 开 | 固定物理缩放；两组 value normalization 均关 |
| KL 控制 | 更新前近似 KL 超过0.2退出剩余更新；注释说明默认阈值刻意偏高 | 自适应 LR，不是强制回滚 | 校准组更新后精确固定方差 KL>0.01回滚 Actor |
| 每个形态预算 | 配置5M；取整实际4,997,120次交互 | 2,097,152次，包括Critic预热 | 两组各65,536次Actor阶段交互；校准收集额外计时 |

不能直接抄 LR 和 std：两边动作尺度、奖励、网络、接触任务均不同。尤其不能把 DERL std=1 搬到当前已握稳的手上。

**可借鉴的优先级：**

1. 内层 PPO 必须先在固定形态上可重复学好，再用其分数选择结构。否则会淘汰“控制器没学好”的好结构。
2. 比较 16 与 64/128 步 rollout，保持总 batch/交互预算接近；更长片段减少频繁依赖边界价值估计。GAE lambda=0.95 仍衰减长期信息，因此延长片段不是终点奖励的万能修复。
3. 保留 Critic 有效学习率；BC 导入后先校准当前策略价值，再允许 Actor 更新。DERL 没有此接续阶段，不能拿其从头训练成功直接证明我们的接续正确。
4. 在稳定接续后逐渐增大、按动作维度设定探索，并允许其学习；不靠永久微小固定噪声期待大幅适应新结构。
5. 增加完整回合成功率、最后保持时长和安全指标，保留最佳 checkpoint。不要仅按某次训练回报决定形态优劣。
6. 小网络可作独立吞吐/学习对照，但改变网络破坏旧 BC 直接继承，不与本次接续修改混在一起。

源码：[参数](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/config.py)、[PPO](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/algos/ppo/ppo.py)、[高斯策略](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/algos/ppo/model.py)、[归一化](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/vec_env/vec_normalize.py)。

## 3. 算力与速度

论文报告：16 台 × 72 CPU = 1152 CPU；288 个形态训练 worker 并行，每个分配4 CPU；累计4000形态，每形态约500万交互，整次搜索少于16小时。约200亿次交互、平均约10代谱系深度。仓库默认 USE_GPU=False、每节点18个worker，与16节点的288worker一致。源码32环境按每进程4环境分组，产生8个环境子进程/worker；子进程数不等于CPU核数，不能据此改写论文的4CPU/worker。

我们上轮8个体计划总预算约1678万交互，实际7个完成约1468万；计划总量约为 DERL 单次搜索的1/1192。每形态训练预算约为其42%，且包含Critic预热。DERL的快主要来自形态级大规模异步并行，并非一个特殊PPO技巧可以把单卡接触任务加速千倍。

单4090应先完成固定基准和少量变异的可靠对照。当前32环境诊断速度约300交互/s，不应线性外推正式256环境吞吐。未来硬件需求应按实际测得的每GPU有效交互/s、需要评估的形态数、每形态预算、随机种子数估算：

`训练时间 ≈ 形态数 × 每形态交互数 × seed数 / (GPU数 × 实测每GPU交互/s × 并行效率)`。

另加校准、加载、评估和失败重试时间。多GPU优先把不同个体分配到不同GPU，不要求为一个小策略做分布式梯度同步。16/32/64活跃个体规模可以随吞吐逐级增加；576不是必须复现的常数。若沿用8基因3档，6561是离散组合上界，不是必须全部训练；先排除不可实例化/严重物理失稳，再做异步小锦标赛和分阶段预算。

论文：[Embodied intelligence via learning and evolution](https://www.nature.com/articles/s41467-021-25874-z)。仓库：[正式ft配置](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/ft.yml)、[README](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/README.md)。

## 4. 任务与碰撞：不能照搬的地方

DERL 的 manipulation.py 使用“靠近物体，再推向目标”的距离进展奖励。正式 mvt.yml 为20cm立方体、目标成功距离0.75m；不是我们的自由物体持续包绕抓握。它对稳定摩擦支撑、局部滑移和终点姿态的要求不同。

形态采用受限几何和离散变异操作（grow_limb、delete_limb、limb_params、density、dof、joint_angle、gear）。生长时检查新碰撞；保存时对初始永久接触生成碰撞排除。这说明结构生成器必须兼容物理引擎，但不意味着可以关闭手指间或手物接触来消除异常。我们的检查应覆盖运行时关节轴、零位穿透、质量惯量、驱动负载和接触尺度；任务困难不等于结构非法。

源码：[操作任务](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/tasks/manipulation.py)、[mvt配置](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/configs/evo/mvt.yml)、[形态与碰撞](https://github.com/agrimgupta92/derl/blob/91e077e355b27bb74fa920c708d606ff9f4d611e/derl/envs/morphology.py)。
