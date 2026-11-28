# 从零运行 Newton 场景

这个入口不需要原实验机、原 Evolution_PC 目录或 IsaacLab 安装。需要 Linux、Python 3.12；AX50 软体场景需要 NVIDIA CUDA GPU 和兼容驱动。基础刚体示例使用 CPU。C 编译器仅供少数自定义渲染器使用。

## 安装

```bash
git clone https://github.com/Jiahe-Zhao/Evolution-Hand.git
cd Evolution-Hand
python3.12 -m venv .venv
.venv/bin/python scripts/bootstrap_scenes.py --install
```

Bootstrap 会验证 AX50 ZIP、场景分卷、每个输入文件和固定动作种子，然后解压到本仓库 `outputs/`。不下载任何私人资产；Python 依赖从包索引安装。模型输入已经纳入普通 Git，不依赖 LFS。首次克隆体积较大，场景输入压缩后约 324 MB。

重复执行 bootstrap 不会覆盖已有实验修改；如果提示输入已被改写，应使用新 checkout 验证。普通运行直接执行下方命令即可。

## 运行

```bash
.venv/bin/python scripts/run_scene.py --list
.venv/bin/python scripts/run_scene.py 04
.venv/bin/python scripts/run_scene.py 04 --mode demo
.venv/bin/python scripts/run_scene.py ax50-04 --side left
.venv/bin/python scripts/run_scene.py ax50-04 --side right
```

- `smoke` 是默认模式：实际模型加载、短程 Newton 物理推进；AX50 同时检查状态有限和 reset。
- `demo` 运行基础手的完整固定动作示例。进程正常结束不等于操作目标成功；任务仍按原判据输出分数。CPU 单场景回放不等同于历史 CUDA 并行实验。
- AX50 的 8 个入口提供左右手装配及短程运行；其任务控制器仍在 `scripts/nov20/ax50_*` 中。不要把能启动误认为抓取/敲击已成功，也不据此启动训练。
- 03 搬运、18 持机点选是历史任务；21 倾倒、22 夹具补充保留。
- 所有标准命令由 `configs/newton_scenes.json` 显式登记。历史参数搜索、回放和报告脚本保留，但它们可能需要先运行相应试次生成数据；不是每个研究脚本都应直接单独执行。

运行可从其他工作目录调用绝对路径的 `scripts/run_scene.py`；仓库根目录由脚本位置确定。标准入口为独立进程，避免不同场景修改模块级控制器后互相干扰。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -p test_portable_assets.py
.venv/bin/python scripts/validate_scenes.py --group rigid
.venv/bin/python scripts/validate_scenes.py --group ax50
.venv/bin/python scripts/validate_scenes.py --group demo
```

输出位于 `outputs/portable_validation/`。基础场景 22 组，AX50 八场景乘左右手 16 组。AX50 短测默认 400 个 0.125 ms 子步，即 0.05 s；这只是启动验收。基础完整动作示例另行跑完，不以烟雾测试替代。

本轮会从 Git 导出的空目录重新解压、创建独立环境，并用文件访问审计阻止读取旧 checkout。实测汇总见 `docs/evidence/portable_release/`。

## 模型与文件

- `assets/releases/AX50_VisualSkin_v3.0.0.zip`：已验收 AX50 包，字节保持不变。
- `assets/releases/nov20/`：场景输入的分卷 ZIP、逐文件清单及 SHA-256；bootstrap 自动拼接，无需手动解压分卷。
- `assets/scene_seed_overrides/`：需要显式命名的初始化姿态/固定动作种子。
- `models/legacy_spread_20261128/`：旧骨手的展开机构模型，含匹配左右手、12 个新增网格及其余依赖。不是 AX50，不替换默认训练手。
- `assets/fonts/`：渲染用字体及其许可证；ffmpeg 可使用 imageio-ffmpeg 包自带程序。
- `scripts/nov20/`：场景构建、控制、评测和研究工具。
- `docs/review/UNCOMMITTED_47.md`：47 处修改的逐项处理；39 项旧 IsaacLab/PhysX 变更作为未应用补丁保留，不冒充已验证。

## 修复范围

移除了场景运行代码中的实验机项目路径；XML 拼装会解析源文件的网格引用，避免移动子树后相对路径失效；补齐模型、网格和固定种子；敲击装配使用其原本定义的手掌朝向。没有改变 AX50 骨长、软体材料或碰撞配置。

新安装环境的版本由 `requirements-scenes.txt` 固定。旧 IsaacLab 训练链是独立可选系统，本发布不包含旧策略 checkpoint，也不将其 Physics/训练修改启用为默认。当前入口验证的是原生 Newton 场景。
