# AX50 VisualSkin v3.0.0 导入

发布包：`assets/releases/AX50_VisualSkin_v3.0.0.zip`。它是 2026-11-21 已验收包的逐字节副本，不包含此后未验收的实验修改。

## 获取

```bash
git clone https://github.com/Jiahe-Zhao/Evolution-Hand.git
cd Evolution-Hand
python3 scripts/unpack_ax50.py
```

也可以 Download ZIP 后解压仓库，再运行同一命令。不要只下载单个 Python 文件。无需 Git LFS、子模块或旧实验目录。

仅检查下载完整性：`python3 scripts/unpack_ax50.py --verify-only`。
指定解压目录：`python3 scripts/unpack_ax50.py --dest /your/assets`。

## 运行

解压后的 `README.md` 是完整说明；`requirements-tested.txt` 记录实际依赖。已验证 Linux / Python 3.12 / CUDA GPU，Warp 1.17.0，MuJoCo/MJWarp 3.12.0。包内包含局部补丁 Newton 1.6.1。普通 macOS 可以解压和检查文件，但此次物理验收使用远端 Linux CUDA 环境。

在匹配依赖环境中：

```bash
python -m pip install --no-deps local_assets/ax50_visualskin_v3_0_0/dist/ax50_visualskin-3.0.0-py3-none-any.whl
python local_assets/ax50_visualskin_v3_0_0/examples/basic.py --side left --wrist locked
python local_assets/ax50_visualskin_v3_0_0/examples/basic.py --side right --wrist six_axis
```

其他例程：`examples/scene.py` 加入道具；`examples/usd_visual.py` 更新 USD 显示。`media/bilateral_motion.mp4` 是实际 Newton 状态的左右手同步回放；`ACCEPTANCE.md` 和 `evidence/FINAL_ACCEPTANCE.json` 是验收结论。

## 使用边界

- 通过 `HandRuntime` 启用物理软层和触觉；只加载 USD 不会启用 NPZ 中的软体。
- 每个实例是独立物理世界。同世界双手互碰和机械臂联动仍需共享后端适配。
- 真实软体是软掌和食指三段垫，其余四指/鱼际保留原刚体覆盖。连续显示皮肤不等于物理软体变形。
- 本轮验收使用 20,000 N/m 探针接触配置；高刚度、高载荷和训练任务未据此认证。
- 原食指软垫保留 6 条非流形边；详见包内已知限制。

SHA-256：`47084259f9b76ae57d3da3c6af8f809c3fddf313ce6de6e13e2efa0767c2caca`。
