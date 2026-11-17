# ax50：Newton 匹配左右手样机

本目录冻结 2026-11-17 完成基础验收的 ax50。用于手部控制、接触测量与小规模任务实验；不替换项目原有训练基准。左右手各自独立仿真，并非双手同时操纵同一个球。

## 模型

- [左手 URDF](models/ax50_left.urdf) 和 [右手 URDF](models/ax50_right.urdf) 是空手模型，不含测试球。骨长固定；五指共 14 个屈曲关节，另有掌骨拱掌、拇指根部横摆和轴向旋转。
- 掌侧与指腹包覆厚度 3 mm；掌骨背侧 1.55 mm，指骨背侧 2.15 mm。指套为随骨节运动的碰撞外壳；掌面形变和大鱼际双支撑由运行时代码计算，不能只加载 URDF 就认为已装载完整软掌。
- 相对上一版样机，ax50 的拇指根部横摆轴向掌面法线转 50°，并补偿第一掌骨安装朝向，以兼顾张开和旋前。实际使用的局部轴、安装旋转向量见 [轴参数](runtime/ax50_axis.txt)、[安装参数（度）](runtime/ax50_mount.txt)。右手的关节、惯量、皮套和球位经物理镜像生成。
- [基础测试](runtime/basic_check.py)、[左右手固定球测试](runtime/matched_trial.py)、[单手自由球测试](runtime/hold30_newton.py) 装载模板并补齐碰撞与掌面力学。测试中腕部固定，未验收自由腕或任务训练接口。

## 验收结果

| 功能 | 实测 | 证据 |
|---|---|---|
| 左右手逐指屈伸 | 每指单独屈曲至 30° 并回位；未驱动屈曲关节串动小于 0.001°，无意外接触 | [左手](evidence/basic_left_summary.json)、[右手](evidence/basic_right_summary.json) |
| 空手对掌 | 两手均完成张开、旋前、拇食指轻触和松开。保持约 0.0734 N，松开后为零 | [明细](evidence/基础功能检查.md) |
| 固定球上下对夹 | 40 mm、35 g；两手各自持续通过 1.608 s。保持段拇指向上约 0.291 N，食指和中指分别向下约 0.271 N、0.290 N | [并排视频](evidence/matched_pinch.mp4)、[力曲线](evidence/matched_pinch_forces.png) |
| 单手自由球留置 | 6.0 s 撤夹具，至 36.0 s 球心最大偏移 0.042 mm；平均向上合力 0.343 N。承担保持的是中指、环指、小指，不能据此判定拇指对掌自由抓稳 | [视频](evidence/hold30_ball.mp4)、[力曲线](evidence/hold30_all_forces.png) |

## 复现

在 Linux + CUDA 的 Newton 环境中运行。需 Newton、Warp、MuJoCo/MJWarp、NumPy、SciPy 和 C 编译器。设置 `NEWTON_PYTHON` 为对应环境的 Python 路径（默认 python3），然后在本目录执行：

```sh
./run.sh basic left
./run.sh basic right
./run.sh matched left
./run.sh matched right
./run.sh hold30
```

入口在临时运行目录复制输入并从 layer.c 编译软层算子，保留该次输出并打印路径。固定球测试生成含夹具的 URDF；自由球与基础测试不能相互替代。证据仅覆盖所列动作与球位，不代表四任务 PPO 或进化训练已通过。

复测环境见 [verified-versions.txt](verified-versions.txt)。
