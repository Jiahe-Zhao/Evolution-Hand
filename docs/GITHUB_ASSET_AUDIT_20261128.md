# GitHub 资产完整性审计（2026-11-28）

## 后续补齐（同日）

本页下方保留第一次审计快照。后续已合入场景进展分支及其包含的 v5 开发提交，整理 447 个场景 Python 脚本、1,714 个场景输入和 3 个显式命名种子，补齐模型、网格、字体与独立运行环境。场景输入采用普通 Git 分卷 ZIP（约 324 MB），无需实验机目录或 Git LFS。

47 处修改已逐项审查：6 项通用修复合入，2 个展开机构模型独立发布，39 项旧训练链修改保留为未应用补丁。原实验目录未覆盖，AX50 v3 原包不变。

当前安装入口见 [从零运行 Newton 场景](PORTABLE_NEWTON_SCENES.md)，实测结果见 [发布证据](evidence/portable_release/)。此后续发布覆盖下方“尚未整合”中关于场景依赖的历史状态；未包含全部历史训练结果、checkpoint 和缓存。

## 第一次审计结论

实验机目录并未全部同步到 GitHub。以前交付的本地压缩包和路径不表示 GitHub 已发布；普通下载默认分支确实拿不到 AX50。

本次只发布已验收的 AX50 VisualSkin v3.0.0 完整包、获取说明和校验解压入口。不会把未经整理的全部实验文件作为稳定版本发布。

## 审计时的状态

仓库：Jiahe-Zhao/Evolution-Hand。修改前 main：`36bd66ab77574d73eac6f0707ea8c8362f2b6314`。

| 内容 | 审计时位置 / 状态 |
|---|---|
| main 默认分支 | 没有 AX50 / VisualSkin 文件 |
| 旧 AX50 原型 | 已在 `codex/ax50-newton-prototype`，提交 `8ad40ef`，比 main 多 1 个提交；含 `prototypes/ax50/` |
| AX50 场景进展 | 已在 `codex/ax50-scenes-progress-20261125`，提交 `6b7b81a`，比 main 多 23 个提交 |
| VisualSkin v3.0.0 完整包 | 原来仅在实验机 `outputs/ax50_visualskin_v3_0_0/dist/` 和本地交付目录；没有 GitHub Release |
| 实验机当前开发分支 | `codex/newton-hold-ppo-20261113`，HEAD `679d09e`，比 main 多 12 个提交；无同名远端分支，但这些提交已在 `codex/v5-tactile-acceptance` 和场景分支发布 |
| 未提交工作 | 47 个已跟踪文件修改、342 个未跟踪条目（含目录；不是文件总数）；另有被忽略输出 |

计数为修复前只读快照，不代表所有未提交内容都适合发布。旧原型、VisualSkin 和后续场景实验不是同一套资产。

## 为什么会缺失

1. `.gitignore` 忽略整个 `outputs/`，新生成的资产不会自动进入 Git。
2. 开发分支没有合入 main。默认 clone / Download ZIP 只拿到所选分支。
3. Git 仓库内容、实验机工作目录和单独交付 ZIP 是三件事；之前缺少统一资产入口。
4. 近期场景脚本可能仍依赖实验目录和额外资产，不能只把脚本复制到 main 就声称可复现。

## 本次补齐

- `assets/releases/AX50_VisualSkin_v3.0.0.zip`：47,158,654 字节，普通 Git 二进制文件，不是 LFS 指针或外链。
- `assets/releases/ax50_visualskin_v3.0.0.json`：固定版本和 SHA-256。
- `scripts/unpack_ax50.py`：校验 ZIP 和 967 个交付文件，默认解压到 `local_assets/`。
- `docs/AX50_ASSET_IMPORT.md`：运行依赖、示例与已验证边界。
- 根 README 添加醒目入口；解压副本被忽略，正式 ZIP 被跟踪。

## 尚未整合

旧原型分支、近期场景分支、12 个已在其他分支发布的 v5 开发提交、47 处未提交修改和其他实验输出没有在这次资产发布中统一合入。训练 checkpoint、缓存和完整日志也不属于本次资产包。不能据此声称“所有内容都已上传”。

需要复现近期场景时，应指定场景编号、对应分支和依赖版本，单独补齐运行资产；本包解决的是独立 AX50 VisualSkin 导入。

## 补充依赖排查

DeepSeek 辅助只读检查，并核对远端分支后确认：当前 HEAD 的 12 个提交不是丢失，而是尚未合入 main；`codex/v5-tactile-acceptance` 与当前 HEAD 均为 `679d09e`。

AX50 场景分支明确不是独立发行包，部分脚本仍从 `outputs/ax50_visualskin_v3_0_0` 和 `outputs/nov20` 取依赖。当前资产包补齐了前者的可获取性，但没有发布所有场景输出。不要把本次补包理解为所有场景已可直接运行。

其他已发现问题包括两个本地 URDF 修改引用了 12 个未跟踪的展开机构 STL，以及近期场景脚本/文档继续在工作区更新。上述内容需单独完成依赖整理和复测，未混入此已验收资产包。
