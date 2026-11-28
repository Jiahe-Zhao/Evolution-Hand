# AX50 任务13（活动提梁）验收 / 渲染辅助说明

本目录新增两个只读辅助脚本，均不启动物理仿真、不改物体/手/运行器：

- `audit_handle.py`：从已保存的 `{tag}.npz` 与 `{tag}_trace.csv` 重算验收结论。
- `render_handle.py`：按记录的 body 姿态重放并输出视频与曲线图，结论直接调用 `audit_handle.audit`。

调用方式（在仓库根目录或本目录均可）：

```bash
python scripts/nov20/ax50_handle13/audit_handle.py <tag>
python scripts/nov20/ax50_handle13/render_handle.py <tag>
```

`tag` 缺省为 `handle_v1`；手侧用环境变量 `AX50_SIDE`（默认 `left`）。`audit_handle.py` 通过时退出码 0，否则 1，并写 `outputs/nov20/ax50_handle13/{tag}_result.json`。

## 输入

- `outputs/nov20/ax50_handle13/{tag}.npz`：`time`、`body_q`（T×N×7，pos+quat）、`body_names`、`particles`、`tri`、`targets`；约定最后两个 body 为 `item`、`bail`。
- `outputs/nov20/ax50_handle13/{tag}_trace.csv`：`time_s, phase, hinge_rad, lift_m, tilt_deg, body_contact_N, bail_contact_N, finite, substeps, qualified_substeps`。
- `outputs/nov20/ax50_handle13/object.xml`：渲染几何来源（仅 `item`、`bail` 两个 body 的 box；floor 平面不参与验收）。

`targets` 只记录、不参与任何判定；IK 目标或导入结果不能算通过。

## 验收判据与实现位置

所有指标都从 `body_q` 重算，`trace.csv` 只做交叉校验（`audit_handle.py` 中 `integrity`）。

| 判据 | 实现 |
| --- | --- |
| 提梁相对器身转起 ≥45° | `R_item^T R_bail` 绕 object.xml 铰链轴 `(0,-1,0)` 的有符号角，`unwrap` 后取度；阈值 `CRIT['hinge_deg']` |
| 器身相对初态升高 ≥0.025 m | `item` body 世界 z 减去 **npz 第 0 帧** z；阈值 `CRIT['lift_m']` |
| 器身倾斜 ≤15° | `arccos(R_item[2,2])`，与 `scripts/nov20/env.py` 的 tilt 定义一致（器身 z 轴偏离世界竖直） |
| 手不得接触器身（≤0.0005 N） | 每帧 `body_contact_N`，且全局最大值也必须 ≤ 0.0005 N（对应 env.py 的 `bad` 锁存语义） |
| 仅经提梁载荷 >0.01 N | 每帧 `bail_contact_N > 0.01` |
| 连续 ≥0.5 s | 每帧同时满足上述条件 + 子步/有限性，取最长连续区间；要求时长 ≥0.5 s 且相邻帧间隔 ≤1.5×中位 dt |
| 逐子步 `qualified_substeps == substeps` 且 >0 | 每帧校验，且要求所有帧都满足 |
| 有限 | trace 的 `finite` 全为真，且 npz `body_q` 全部有限 |
| 初态 | npz 第 0 帧（第一帧仿真记录），不随手动作移动 |

交叉校验容差：时间 1e-6 s、提梁角 0.25°、倾斜 0.25°、升高 0.00025 m；同时要求帧数一致、时间递增、最后两个 body 为 `item,bail`。

关于升高基线：判据使用 npz 第 0 帧，而 trace 的 `lift_m` 以仿真前 `z0` 为基线，二者可能相差一个常量。`audit_handle.py` 用 trace 第一帧的 `lift_m` 作为该常量做交叉校验，并把差值写入 `integrity.lift_origin_offset_m`；判定本身仍用第 0 帧基线。

若存在 `{tag}_status.json`，还要求 `stop == "completed"`；文件不存在则不作为失败条件。

## 渲染

- 几何：`object.xml` 原始 box（含 `item` 各壁、`bail` 的 arm0/arm1/grip），用 MuJoCo 在 CPU 解析；逐帧按 `body_q` 的 `R_b @ R_local`、`p_b + R_b @ p_local` 变换，不插值、不编造。
- 手：AX50 `VisualSkin`（`device='cpu'`）；软体：npz 实际保存的 `particles` / `tri`。
- 两个固定视角：视角1 看器身底部与地面净距（地面方片 + 底部最低点到地面的竖条 + 净距数值），视角2 看提梁铰链（铰链红标 + 转角数值）。相机按实际点位自动取景。
- 采样 30 Hz、编码 15 fps，即 2x 慢放；复用 `smooth_render.panel`、`render_core.strip/encode`。
- 中文：优先 Noto Sans CJK 等本机字体，并用 PUA 字形对比检测缺字；任一字体缺字则整体回退英文文案，避免豆腐块。
- 输出：`{tag}.mp4`、`{tag}_curves.png`（提梁角 / 升高 / 倾斜 / 接触力四联图，合格区间绿色阴影）、`{tag}_render.json`。视频标题中的通过/未通过直接来自 `audit_handle` 的计算结果。

## 当前状态（重要）

- 本目录当前**没有**任何 `{tag}.npz`，因此两个脚本都**未实际运行**，未生成视频、曲线或验收结论；只做了 `py_compile` 语法检查（通过）。
- 因此不能据此声称任务通过；拿到实际 `{tag}.npz` 与 `{tag}_trace.csv` 后，先跑 `audit_handle.py`，再跑 `render_handle.py`。
- 本次只新增 `audit_handle.py`、`render_handle.py`、`SUPPORT_NOTES.md` 三个文件，未改动物体、手、运行器、共享代码或远端文档。
