# 腕部 Roll/Pitch/Yaw 更新（2026-11-05）

当前机械手在原有 21 个手指关节和 XYZ 平移腕部基础上，新增三个串联旋转关节：`wrist_roll`、`wrist_pitch`、`wrist_yaw`。

参数：
- 平移腕部：XYZ，各 `[-0.25, 0.25] m`，最大速度 `0.02 m/s`，驱动力 `100 N`。
- 旋转腕部：Roll/Pitch/Yaw，各 `[-pi/2, pi/2] rad`，最大速度 `2 rad/s`，驱动力矩 `20 N·m`。
- 新增移动质量：`0.3 kg`。
- 旋转轴在原手根部局部坐标系定义。
- 腕部不进入形态基因，避免改变手指形态搜索空间。

验证入口：`test_motion.py`。视频输出：`runtime/wrist_rpy_v2/test4/motion.mp4`。当前验证范围为全手运动与导入稳定性，不代表抓握或 Strike 成功。
