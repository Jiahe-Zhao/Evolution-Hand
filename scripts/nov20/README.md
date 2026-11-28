# Portable entrypoints

Use the repository-level `scripts/bootstrap_scenes.py` and `scripts/run_scene.py` first. See [installation and validation](../../docs/PORTABLE_NEWTON_SCENES.md). No original experiment checkout or IsaacLab installation is needed for these native Newton scenes. Historical optimizer/report scripts below may require data produced by the corresponding run.

# Newton 手任务：搬运、掌内换面、活动提梁、拇指点选
在 Evolution_PC 下运行，Python 使用本仓库 .venv/bin/python。依赖 hand_111 左手 URDF 与相邻 meshes。使用 Newton SolverMuJoCo；--gpu 启用 MJWarp CUDA。直接 Newton 接口，不修改现有 v5 训练入口或权重。

    python scripts/nov20/run_final.py stone --gpu
    python scripts/nov20/run_final.py stone negative --gpu
任务名：carry / stone / handle / phone。negative 保持初始关节目标，无任务动作。所有试次物体在重力下自由运动；被动提梁不受控制。资产初态和目标在 build.py，验收逻辑在 env.py，动作脚本在 run_final.py。每步仅写手指与腕关节位置目标。dt=1/240 s。NPZ 保存每步动作和判据，刚体位姿每 8 步一帧；视频用同一组仿真关节和自由物体位姿以 30 fps 渲染。

Gymnasium 用法：将 scripts/nov20 加入 PYTHONPATH，import gym_tasks 后使用 gymnasium.make("EvolutionNewton-StoneReorient-v0")。另有 Carry / BailLift / ThumbTap 三个 ID。reset 返回刚体世界位姿矩阵（xyz + xyzw）以及动作关节名；step 输入关节目标（米、弧度）。单环境、固定初态、稀疏终局奖励；这里不提供已训练策略。按任务定义使用交互物体与手部碰撞，不用于手的自碰撞评估。几何预览呈现实际用于测试的刚体代理。
