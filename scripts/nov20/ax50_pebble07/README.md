# AX50 场景07（石子）资产检查与适配准备

范围仅限资产提取与 CPU 导入/静态检查。本目录不产生抓取结论：**能导入不等于操作通过**。

## 命令

使用 `/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python`（Python 3.12.14，Warp 1.17.0）。

```bash
# 1. 从 outputs/nov20/pebble.xml 提取原盘+三石，生成 object.xml 与审计 JSON（CPU）
python scripts/nov20/ax50_pebble07/inspect_asset.py

# 2. CPU 静态检查 object.xml 的物体/自由关节/质量/无驱动（不实例化 runtime）
python scripts/nov20/ax50_pebble07/pebble_runtime.py

# 3. 可选：实例化 AX50 共享世界并前进 0.05 s（需要 GPU；本次未运行）
python scripts/nov20/ax50_pebble07/pebble_runtime.py --smoke
```

产物（`outputs/nov20/ax50_pebble07/`）：

- `object.xml`：清理后的场景（原盘 + 三石）。
- `asset_audit.json`：删除项、保留项签名、作者质量/摩擦、导入结果与检查。
- `static_check.json`：CPU 静态检查结果。
- `shared_smoke.json`：仅 `--smoke` 时生成，本次不存在。

## 发现

- 源场景 `outputs/nov20/pebble.xml`（70195 B）为旧左手场景：1 个 `diagnostic_world`
  （34 body / 102 geom / 27 joint / 102 STL mesh）、27 个 actuator、225 个接触对。
- 提取后保留：
  - worldbody：`floor`、`light`、`tray`、4 个 `rim_*`、`item`、`neighbor1`、`neighbor2`；
  - 3 个 inline mesh：`peb0`、`peb1`、`peb2`；
  - 21 个对象接触对（每石 foot/tray、stone/tray、stone/4 rim，另加 stone1-stone0、
    stone2-stone0、stone2-stone1）。
- 删除：34 个手 body、102 个手 geom、102 个 STL mesh、204 个手接触对、27 个 actuator。
- 作者质量（body 内 geom mass 求和）与 Newton CPU 导入质量一致：

  | body | 作者质量 (kg) | 导入质量 (kg) |
  |---|---|---|
  | item | 0.002110 | 0.0021100000012665987 |
  | neighbor1 | 0.0013457176 | 0.0013457175809890032 |
  | neighbor2 | 0.0010065592 | 0.0010065592359751463 |

- 三石各保留 1 个 `freejoint`（`free0/free1/free2`），导入为 3 个 FREE 关节，
  18 DOF、`joint_q` 长度 21；所有对象 DOF 的 `target_ke/kd` 为 0、target mode 为 NONE。
- 导入后 12 个 shape，摩擦首系数：`floor` 1.0，其余（stone/foot/tray/rim）0.85，
  与源 XML 一致（float32 误差 < 1e-6）。
- 保真说明：源 `<pair>` 的 `solref/friction` 已原样写入 `object.xml`，但 Newton 刚体接触
  按 shape 建模，导入器不会按 pair 应用这两项；shape 摩擦正常导入。这是已知差异，不是本
  次引入的改动。

## 适配方式

`pebble_runtime.py` 的 `PebbleRuntime(HandleRuntime)` 复用 `ax50_handle13/handle_runtime.py`
的架构：保留手腕 PD、palm 局部阻尼修复和碰撞过滤 pair 的表示恢复；仅把物体 XML 换成
`object.xml`，按短名映射 `item/neighbor1/neighbor2` 与 `free0/free1/free2`，并关闭对象 DOF
驱动。未修改物体尺寸，未新增全局禁碰。`_add_brush` 已用 CPU 哑元 builder 单独验证
（body 映射、自由关节、驱动清零）。

## 依赖

- `/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python`
- `newton`（site-packages）、`warp`
- `ax50_visualskin_v3_0_0`、`scripts/nov20/ax50_keys17/damped_adapter.py`
- `scripts/nov20/ax50_handle13/handle_runtime.py`

## 未完成项

- 未运行 GPU smoke（`--smoke` 命令已提供）。
- 未验证 AX50 双指捏取、抬升 ≥0.020 m、邻石位移 ≤0.005 m 等操作指标。
- 未修改评分、旧模型、公共运行时或文档；`docs/TASK07_AX50_PEBBLE.md` 的状态保持原样。
- 因此本目录当前只能说明“资产可提取、可 CPU 导入、接入代码路径已接通”，不能说明任务通过。
