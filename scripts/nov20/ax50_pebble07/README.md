# AX50 场景07：捏取石子

## 2026-11-25 接触处理与捏取启动

状态：接触参数覆盖已实测通过；顺序屈指预备动作完成。尚未实际夹起目标石，不计07通过。

- 原运行时实际保留225个显式接触对；AX50采用Newton接触后，原21个对象接触对虽在模型中，参数没有自动用于外部接触。新增仅scene07的实例级GPU覆盖，在转换接触后、求解前按MuJoCo geom ID写入原pair friction/solref/solimp/condim。原204个旧手规则统一映射为246个AX50有效刚体geom与目标石两形状的492条规则；加原21条，共513条。没有统一提高所有物体摩擦，没有修改公共Newton、物体形状、质量或碰撞过滤。
- 实际GPU接触验证通过：foot/tray摩擦[1,1,.003,.0001,.0001]、solref[.009,1]；stone/tray摩擦[.9,.9,.003,.0001,.0001]、solref[.01,1]。石体接触测试采用单独旋转石子的诊断初态，未修改原任务初态，不计捏取。手与石的规则映射完成，真实手石接触仍待后续验证。软垫保留其原耦合参数，不宣称等价旧刚性手。
- 非流形警告来自AX50 pads四面体表面：6条边各有4个相邻面。逐条确认模型保留全部边、顶点到边邻接、4个面，且6条边弯曲刚度和阻尼均0；不存在因任取两面而产生的弯曲力。本次不重网格、不删接触、不屏蔽警告；警告仍可见，表面拓扑本身仍非流形，后续接触试验继续关注该区域。
- 修复开启后完成0.9秒空中拇指先摆姿、食指后屈曲及保持，100 Hz记录状态均有限，末态最大关节目标误差0.000383 rad（0.0219°），最大软粒子速度0.1956 m/s；两邻石全程采样最大位移0.1287/0.1007 mm。手远离物体，以上不是夹持或任务验收。
- CPU首个对捏候选间距20.303 mm，但掌弓、拇指转动靠近限位，未执行下降。下一步重新选择实际指腹接触点和无顶限位的对捏姿态，再垂直接近、夹持和提起。AX50拇指末节为link_1_2，食指末节link_2_3；必须按解剖末节映射原标准，不能机械复用旧手拇指link_1_3名字。

证据目录outputs/nov20/ax50_pebble07：pair_verification.json、verified_topology.json、preshape_fixed_result.json、preshape_fixed_trace.csv、preshape_fixed.npz、pinch_candidate.json。GPU对照切换开关时须清空_advance_graph，防止CUDA图保留旧开关值。

## 运行
在/home/zjh/Evolution_PC使用/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python：
- scripts/nov20/ax50_pebble07/inspect_asset.py：原对象提取与CPU核对。
- scripts/nov20/ax50_pebble07/verify_pairs.py：实际GPU接触参数对照。
- scripts/nov20/ax50_pebble07/verify_topology.py：六条边、零弯曲材料、表面及顶点邻接复核。
- scripts/nov20/ax50_pebble07/run_preshape.py：修复开启后的顺序屈指；RUN_TAG控制输出名。
- scripts/nov20/ax50_pebble07/fit_pinch.py：仅CPU候选，不可直接视为可执行轨迹。

需远端AX50 v3外部资产及原场景；不是独立发行包。旧早期检查记录为历史，最新状态以上文为准。
