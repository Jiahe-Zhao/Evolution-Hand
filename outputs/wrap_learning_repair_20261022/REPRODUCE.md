# 复现入口

在远端 `/home/zjh/Evolution_PC/outputs/wrap_learning_repair_20261022` 运行，Python 使用 `/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python`。先确认 `reproducibility_manifest.json` 中的资产与数据存在且哈希一致。所有输出目录必须是新目录。

## 已验证的 BC29 闭环评估

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python repair_learning.py \
  --headless --plans ../contact_control_calibration_20261021/plans_wrap_cover_selected.json \
  --pad-face ymixed --kp 3 --kd .3 --contact-k 1500 --hand-gravity \
  --num-envs 64 --smoke --episode-steps 1500 --eval-steps 1500 --push \
  --checkpoint bc_contraction29/bc_initial.pth --mode eval --seed 2027 \
  --output NEW_BC_EVALUATION
```

`--smoke` 这里只适配 64 环境的 minibatch 大小；评估没有梯度更新。`--sample-eval --action-seed 2027` 可检查小幅随机动作。先前已存储的评估使用默认 30 秒截止、实际运行 25 秒；当前命令将截止也对齐 25 秒，不在最后一步重置后覆盖评估状态。

## 从示范重建 BC

沿用上面相同场景参数，按顺序运行三个独立目录：

1. BC01：不传 checkpoint；`--mode bc --num-envs 32 --smoke --bc-steps 8000 --bc-lr .0003 --obs-noise .0005 --seed 42`。
2. BC20：加载 BC01 的 `bc_initial.pth`；增加 `--refit --bc-steps 12000 --bc-lr .00001 --contraction 100`。
3. BC29 是当前场景验证通过的基准：加载 BC20；同样补训，`--contraction 1000`。不能仅按小规模结果选模型。

真实数据从 `../wrap_robustness_20261021` 的审计与轨迹文件载入；训练清单由每次运行的 `dataset.json` 给出。历史结果以各运行的 `controller_snapshot.py` 和 `execution.json` 为准；当前入口包含后续终止处理修复，不承诺跨版本逐位一致。

## 最后一组 PPO 诊断

```bash
bash run_short_diagnostic.sh bc_contraction20/bc_initial.pth NEW_PPO_OUTPUT
```

此命令是实验复现，不是自动批准长训。成功标准：同一组扰动评估不破坏已有抓握，且独立种子结果通过。原多代进化任务不由此命令恢复。

## 证据文件

- `ROOT_CAUSE_AND_RESULTS.md`：根因、失败对照、结果和局限。
- `REUSE_GUIDE.md`：跨场景检查流程。
- `results_summary.json`：逐实验成功率与物理峰值。
- 每个运行的 `execution.json`、`effective_agent.yaml`：命令与实际配置。
- `training_metrics.json`：学习率、KL、回报与轮数。
- `critic_warmup.json`、`residual_contract.json`：参数冻结与实际训练预算断言。
- `reproducibility_manifest.json`：外部依赖和模型哈希。

模型、原始轨迹和仿真资产保留在远端；Git 归档代码和证据，不包含这些大文件。
