# 操作物体建模：捏取石子、刮削表面附着层
本批只更换新增场景的优先级。原掌内换面与活动提梁模型、验收视频保持有效。搬运和手机点选归档，不列入当前重点。

|任务|资产及接触|评分边界|
|---|---|---|
|07 捏取石子|带浅边的硬盘；三颗独立不规则凸体，有不同尺寸、质量、朝向和稳定着地面。中心暖色石为目标，约 20 × 15 mm、2.1 g。|目标被拇指和食指末节同时接触，排除掌托和其他手指支撑；离盘、抬升 ≥20 mm，连续 0.3 s；邻石全程最大三维位移 ≤5 mm。末节碰撞用于判定接触，未细分指腹感受区。|
|08 刮削附着层|自由刮具由握部、过渡肩、工作缘组成；工具从托架开始；固定工件带 64 × 30 × 1.2 mm 的连续涂层，分为 3 × 8 个接触区。|握部有手部接触，工作缘接触涂层，法向合力 0.3–3 N，工具俯仰角 30°±15°，切向速度 5–200 mm/s。单区累计 3 mm 有效刮程才计入，至少 20/24 区达标。颜色反映有效刮程；不模拟厚度变化或碎屑。|

复跑（在 Evolution_PC 根目录，用 IsaacLab3 的 Python）：
- python scripts/nov20/objects_check.py：石子 720 步、刮削 240 步 MJWarp GPU 空动作测试；物体单独预览隐藏手以便检查几何。
- python scripts/nov20/objects_probe.py：独立落下自由刮具，验证工作缘测力且无手持不计分。
- Gymnasium：import gym_tasks 后使用 EvolutionNewton-PebblePinch-v0 或 EvolutionNewton-SurfaceScrape-v0；step 的 info.task_metrics 含任务专属指标。
- 上述检查不等同于手实际捏起石子或完成整片刮削；关节控制实测见文末。

实测：240 Hz，RTX 4090 / Newton MJWarp。空动作 3 秒，目标石沉降 0.18 mm，邻石最大位移 1.09 mm；空动作 1 秒，刮削覆盖 0/24。独立落具探针峰值 7.38 N，无握持，覆盖 0/24。

## 关节动作正例复验

hand_111 左手；原场景和原 Score，Newton/MJWarp CUDA，步长 1/240 s，720 步。两项均从标准初态开始，控制端只写手部关节目标。固定初态的轨迹见证，不是策略成功率。

|任务|实测|判定|
|---|---|---|
|07 捏取石子|最高抬升 30.42 mm；邻石最大位移 1.09 mm；双指捏取、排除旁撑及抬升条件连续成立 165/240 = 0.688 s|通过，首次成功 1.658 s；末端没有持续持稳到回合结束|
|08 刮削附着层|首块完成前，手接触握部且工具不碰托架时，最高抬升 20.64 mm；完成 20/24 区，要求 20 区|通过|

07 为单环境回放。08 为 64 个并行环境中的第 23 号（从 0 起），同一环境、同一条连续轨迹。视频从该次 Newton 仿真的关节状态渲染，显示两机位；没有拼接不同轨迹。使用原 Score 对该环境的工具接触约束、逐接触解算力及工具位姿复算。手接触握部不等于全程无滑脱，未测量去除质量。

- 石子：[操作视频](../outputs/nov20/verified_objects/pebble/operation.mp4)、[逐步指标](../outputs/nov20/verified_objects/pebble/metrics.json)。
- 刮削：[操作视频](../outputs/nov20/verified_objects/scrape/operation.mp4)、[逐步指标](../outputs/nov20/verified_objects/scrape/metrics.json)。
- 刮削原始证据：outputs/nov20/scrape_dwell/record046.npz；同目录 best 文件保存动作。原始计分模块 SHA-256：eddb8098e6cf4144de7c4b2a95aa6323849d7884eb3c370b53bc209da8a1900e。
