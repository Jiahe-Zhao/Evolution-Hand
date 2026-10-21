# 基础测试补完与视频

日期：2026-10-21。本轮未改手结构；正式训练继续暂停。

## 压触问题定位

此前单侧压触稳定在1.20 N，并非足够证据说明指腹承不了2 N。独立台架的求解器关节速度上限设为0.01 m/s，会影响该工况下的接触受力。

保持指腹几何、材料、质量、时间步和位置目标闭合速度1 mm/s不变，仅将求解器速度上限改为1 m/s：

- 位置驱动达到2 N；首次达到后保持目标位置，实测约2.02 N，连续超过2 N达0.5 s。
- 零PD增益的独立力驱动：旧上限下命令2.5 N，实测仅0.0807 N；新上限下命令2.2521 N，实测2.2521 N。

上述对照支持低求解器速度上限是本台架低接触力的主要原因。1 m/s是诊断台架的求解器上限，不是命令运动速度，也不是机械手硬件速度规格。不能据此直接改所有训练关节，更不能据此认定此前Strike弹飞已解决。

## 验收范围

- 完整手导入、21个关节、5个独立指腹材料审计：此前已通过。
- 原/新指尖接触身份对照：轴承先接触改为指腹先接触；CPU逐形状回调验证。
- 修正台架后的2 N压触持续0.5 s：通过。
- 80 g自由物体撤支撑后保持3 s：录像复测通过，最大滑移约0.52 mm，低于预设3 mm阈值。

录像复测的滑移比此前无渲染测试约0.0044 mm更大，不能混用两次结果。最终每个视频对应独立的report.json。本轮未对渲染/执行时序差异进一步归因。

## 视频

实时Isaac相机录制，30 fps，960×720，深色背景。绿色为指腹，橙色为轴承，蓝色为物体。画面显示仿真时间、实际接触力、支撑撤除状态和已观测最大滑移。

远端 `press_dark/test.mp4` 与 `hold_dark/test.mp4`；本地交付 `videos/press_2N.mp4` 与 `videos/hold_80g_final.mp4`。

这些是从真实手模型提取指尖的独立台架测试，尚未覆盖完整手抓取、腕部搬运及自由工具击打；也不等同于硬件标定。

## 复现

在远端本目录执行：

```bash
/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python test_baseline.py --headless --device cuda:0 --record --mode press --source candidate/right/urdf/current_agent.urdf --output press_repeat
/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python test_baseline.py --headless --device cuda:0 --record --mode hold --source candidate/right/urdf/current_agent.urdf --output hold_repeat
```

求解器速度上限默认1 m/s。复现旧限速对照加 `--solver-velocity-limit .01`；力驱动对照加 `--drive effort`。完整结果读取report.json，不以进程退出码作为验收标志。
