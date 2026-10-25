# DERL 对标纯 PPO 验证结果

已完成 9/9，失败 0。未完成运行不计为0成功。

| 形态/seed | 状态 | 最终完整回合成功 | 训练步数 |
|---|---|---|---|
| baseline_s2027 | completed | 0/256 | 4980736 |
| thumb_lateral_plus_s2027 | completed | 91/256 | 4980736 |
| thumb_length_plus_s2027 | completed | 16/256 | 4980736 |
| baseline_s2028 | completed | 0/256 | 4980736 |
| thumb_lateral_plus_s2028 | completed | 0/256 | 4980736 |
| thumb_length_plus_s2028 | completed | 0/256 | 4980736 |
| baseline_s2029 | completed | 0/256 | 4980736 |
| thumb_lateral_plus_s2029 | completed | 0/256 | 4980736 |
| thumb_length_plus_s2029 | completed | 0/256 | 4980736 |

## 判读

只有实际完整回合成功才能证明本次场景下完成过任务。稳定收敛需结合三个seed的成功率与学习曲线，不能以个别成功或训练回报替代。
若固定预算内没有成功，只能说明本配置/预算下未学会；不能证明纯PPO或该结构不可行。
评估从物理有效预抓取姿态开始；没有BC或父策略载入。最终均值策略与训练随机策略分别报告。
实验采用同一验证扰动集合选择best；不是完全独立的最终测试集。

本轮有 2 次运行在最终评估完成至少一个完整任务。

